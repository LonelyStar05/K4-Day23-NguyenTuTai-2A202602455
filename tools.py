"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

ATOM = "{http://www.w3.org/2005/Atom}"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
SUMMARY_CHARS = 600
FETCH_CHARS = 12000
ARXIV_GAP_S = 3.0
HEADERS = {"User-Agent": "deep-research-lab/1.0"}


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0, sleep=time.sleep):
    """Call fn(); when it raises RetryableError, wait and call it again.

    The wait is the server's Retry-After when given, else exponential backoff base * 2**attempt with random jitter;
    both are capped at `cap`. The last failed attempt re-raises without sleeping. Other exceptions are not retried.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                backoff = min(base * 2 ** attempt, cap)
                delay = backoff / 2 + random.uniform(0, backoff / 2)  # "equal jitter": stays within [b/2, b]
            sleep(max(delay, 0.0))


def _retry_after(response):
    """Seconds from a numeric Retry-After header, else None."""
    try:
        return float(response.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def _http_get(url, params, timeout=30.0):
    """One GET; transient failures (network, 429, 5xx) become RetryableError, other HTTP errors raise."""
    try:
        response = httpx.get(url, params=params, headers=HEADERS, timeout=timeout, follow_redirects=True)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {url}", retry_after=_retry_after(response))
    response.raise_for_status()
    return response


class SourceDown(Exception):
    """A source failed every retry recently: answer at once instead of waiting through the retries again."""


_down_until = {}
COOLDOWN_S = 180.0


def _guarded(source, fn):
    """Run fn() unless `source` is cooling down; a call that exhausts its retries starts a cooldown for the source.
    The agent then gets an immediate ERROR and switches to another source instead of waiting minutes per call."""
    if time.monotonic() < _down_until.get(source, 0.0):
        raise SourceDown(f"{source} is rate limited right now; use another source")
    try:
        return fn()
    except RetryableError:
        _down_until[source] = time.monotonic() + COOLDOWN_S
        raise


def _clean(text):
    return " ".join(str(text or "").split())


def _short(text, limit=SUMMARY_CHARS):
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {exc}"


class Throttle:
    """Keeps at least `gap` seconds between two calls to one source, also across parallel researchers (threads)."""

    def __init__(self, gap):
        self.gap = gap
        self._lock = threading.Lock()
        self._last = 0.0

    def call(self, fn):
        with self._lock:
            wait = self.gap - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                return fn()
            finally:
                self._last = time.monotonic()


_arxiv_throttle = Throttle(ARXIV_GAP_S)
_hf_throttle = Throttle(1.0)  # parallel researchers bursting the HF API get HTTP 429


# ---- TODO 2: arXiv ----
def _arxiv_get(params):
    """GET arXiv keeping at least ARXIV_GAP_S seconds between two calls (arXiv API etiquette)."""
    return _arxiv_throttle.call(lambda: _http_get(ARXIV_URL, params))


def _parse_arxiv(xml_text):
    records = []
    for entry in ET.fromstring(xml_text).findall(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
        if "/abs/" not in raw_id:
            continue
        arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1])
        records.append({
            "id": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "published": (entry.findtext(f"{ATOM}published") or "")[:10],
            "title": _clean(entry.findtext(f"{ATOM}title")),
            "summary": _short(entry.findtext(f"{ATOM}summary")),
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by a few keywords (e.g. "world model video"), newest first.
    Use short keyword queries, not sentences. Returns a JSON list of {id, url, published, title, summary}
    (url = https://arxiv.org/abs/<id>), "NO RESULTS", or "ERROR: ..."."""
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query or "")
    terms = [t for t in terms if t.upper() not in {"AND", "OR", "ANDNOT", "ALL", "TI", "ABS"}]
    if not terms:
        return "NO RESULTS"
    params = {
        "search_query": " AND ".join(f"all:{t}" for t in terms),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": _clamp(max_results, 1, 30, 10),
        "start": 0,
    }
    try:
        response = _guarded("arxiv", lambda: with_retry(lambda: _arxiv_get(params), attempts=5, base=3.0, cap=60.0))
        records = _parse_arxiv(response.text)
    except Exception as exc:  # noqa: BLE001 - a tool never raises
        return _error(exc)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- TODO 3: Hugging Face ----
def _hf_record(item):
    """Map a Daily Papers / search item to {id, url, published, title, summary, upvotes, github, stars} (or None)."""
    paper = item.get("paper") or {}
    paper_id = paper.get("id")
    if not paper_id:
        return None
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
        "title": _clean(paper.get("title") or item.get("title")),
        "summary": _short(paper.get("ai_summary") or paper.get("summary") or item.get("summary")),
        "upvotes": paper.get("upvotes", 0) or 0,
        "github": paper.get("githubRepo") or "",
        "stars": paper.get("githubStars", 0) or 0,
    }


def _hf_get(url, params):
    response = _guarded("huggingface",
                        lambda: with_retry(lambda: _hf_throttle.call(lambda: _http_get(url, params)), cap=60.0))
    items = response.json()
    return [r for r in map(_hf_record, items if isinstance(items, list) else []) if r]


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    params = {"limit": _clamp(limit, 1, 100, 30)}
    if date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.strip()):
        params["date"] = date.strip()
    try:
        records = _hf_get(HF_DAILY_URL, params)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    if keyword:
        needle = keyword.lower()
        records = [r for r in records if needle in f"{r['title']} {r['summary']}".lower()]
    records.sort(key=lambda r: r["upvotes"], reverse=True)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic (a few keywords). Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} (url = https://huggingface.co/papers/<id>)."""
    if not (query or "").strip():
        return "NO RESULTS"
    try:
        records = _hf_get(HF_SEARCH_URL, {"q": query.strip(), "limit": _clamp(limit, 1, 50, 10)})
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _exa_key():
    return (os.getenv("EXA_API_KEY") or "").strip()


def _redact(text):
    key = _exa_key()
    return text.replace(key, "***") if key else text


def _is_rate_limited(text):
    return bool(re.search(r"rate[\s_-]?limit", text or "", re.I))


def _parse_mcp(response):
    """The JSON-RPC message of an MCP answer: either plain JSON or server-sent events (`data: {...}` lines)."""
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload:
                    return json.loads(payload)
        raise ValueError("no data line in the MCP event stream")
    return response.json()


def _exa_once(name, arguments):
    key = _exa_key()
    url = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    try:
        response = httpx.post(url, json=body, headers=headers, timeout=60.0)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from Exa", retry_after=_retry_after(response))
    response.raise_for_status()
    message = _parse_mcp(response)
    if "error" in message:
        error_text = str(message["error"].get("message", message["error"]))
        if _is_rate_limited(error_text):
            raise RetryableError("Exa rate limit")
        raise RuntimeError(f"Exa JSON-RPC error: {error_text[:300]}")
    result = message.get("result") or {}
    texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    text = "\n".join(t for t in texts if t).strip()
    # Free tier: HTTP 200 with a notice in the text and a flag in result._meta -> NOT page content, retry instead.
    meta = result.get("_meta") or {}
    meta_flag = any(_is_rate_limited(str(k)) and v for k, v in meta.items()) if isinstance(meta, dict) else False
    if meta_flag or (result.get("isError") and _is_rate_limited(text)):
        raise RetryableError("Exa rate limit")
    if result.get("isError"):
        raise RuntimeError(f"Exa tool error: {text[:300]}")
    return text


def _exa_call(name, arguments):
    return _guarded("exa", lambda: with_retry(lambda: _exa_once(name, arguments), attempts=5, base=5.0, cap=60.0))


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa): blogs, surveys, project pages, docs. Describe the ideal page in natural language
    (e.g. "survey paper reviewing world models for robotics"); `objective` says what you want to learn.
    Returns clean text of the top results, each with its URL, or "NO RESULTS" / "ERROR: ..."."""
    if not (query or "").strip():
        return "NO RESULTS"
    arguments = {
        "query": query.strip(),
        "objective": (objective or "").strip() or f"Find authoritative pages about: {query.strip()}",
        "numResults": _clamp(num_results, 1, 10, 5),
    }
    try:
        text = _exa_call("web_search_exa", arguments)
    except Exception as exc:  # noqa: BLE001
        return _redact(_error(exc))
    return _redact(text) if text else "NO RESULTS"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page or a blog post) as markdown.
    Long pages are truncated to ~12000 characters. Returns the text, "NO RESULTS" or "ERROR: ..."."""
    if not str(url or "").startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    try:
        text = _exa_call("web_fetch_exa", {"urls": [url]})
    except Exception as exc:  # noqa: BLE001
        return _redact(_error(exc))
    if not text:
        return "NO RESULTS"
    text = _redact(text)
    return text if len(text) <= FETCH_CHARS else text[:FETCH_CHARS] + "\n...[truncated]"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")

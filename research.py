"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from agents import (FINALIZER_PATH, NOTES_DIR, RECURSION_LIMIT, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR,
                    build_lead_agent)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


class ToolTrace(BaseCallbackHandler):
    """Prints one line per tool call (lead and subagents) so a long run shows progress instead of silence."""

    def __init__(self):
        self.start = time.monotonic()

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "?"
        print(f"[{time.monotonic() - self.start:6.0f}s] {name}: {str(input_str)[:120]}", file=sys.stderr, flush=True)

    def on_tool_end(self, output, **kwargs):
        text = str(getattr(output, "content", output))
        if text.startswith(("ERROR", "NO RESULTS")):
            print(f"         -> {text[:120]}", file=sys.stderr, flush=True)


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic).lower()).strip("-")
    return slug[:60].strip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n"
            f"Today is {date.today().isoformat()}.\n"
            "Produce the cited survey report following your steps 1-8: plan, delegate at least 3 sub-questions to "
            f"researchers in parallel (notes in {NOTES_DIR}/), merge {SOURCES_PATH}, write {REPORT_PATH}, "
            f"run {FINALIZER_PATH} then {VALIDATOR_PATH} until it prints OK, and spot-check with citation-checker.")


FAMILIES = ("arxiv", "hf-daily", "hf-search", "web")
MIN_FAMILIES = 3
FAMILY_FIX_ROUNDS = 2


def missing_families(backend):
    """Families absent from the sandbox sources.json when it holds fewer than MIN_FAMILIES of them, else []."""
    raw = download(backend, [SOURCES_PATH]).get(SOURCES_PATH)
    try:
        present = {s.get("source") for s in json.loads(raw or b"[]") if isinstance(s, dict)}
    except ValueError:
        present = set()
    return [] if len(present & set(FAMILIES)) >= MIN_FAMILIES else [f for f in FAMILIES if f not in present]


def build_family_fix_prompt(missing):
    """Follow-up message to the lead when the finalized sources cover too few source families."""
    return (f"{SOURCES_PATH} covers fewer than {MIN_FAMILIES} source families. Missing: {', '.join(missing)}. "
            "Delegate one researcher per missing family (hf-daily -> hf_daily_papers with a keyword and, if needed, "
            "a few recent dates; hf-search -> hf_search_papers; arxiv -> arxiv_search; web -> web_search) with a new "
            f"notes file in {NOTES_DIR}/. Then add the new sources to sources.json, cite them where they support the "
            f"report, and redo steps 6-7 (run {FINALIZER_PATH}, then {VALIDATOR_PATH} until OK).")


def _get(message, key, default=None):
    return message.get(key, default) if isinstance(message, dict) else getattr(message, key, default)


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    tool_calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in _get(message, "tool_calls") or []:
            tool_calls[call["name"] if isinstance(call, dict) else call.name] += 1
        usage = _get(message, "usage_metadata") or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": tool_calls.get("task", 0),
        "tool_calls": dict(tool_calls),
        "tokens": tokens,
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    A failed run (no report, empty report, missing or invalid sources.json) raises RuntimeError and writes nothing."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = (files.get(REPORT_PATH) or b"").decode("utf-8", errors="replace")
    if not report.strip():
        raise RuntimeError(f"the agent produced no report ({REPORT_PATH} is missing or empty)")
    raw_sources = files.get(SOURCES_PATH)
    if not raw_sources:
        raise RuntimeError(f"{SOURCES_PATH} is missing")
    try:
        sources = json.loads(raw_sources.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} must be a non-empty JSON list")

    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({str(s.get("source")) for s in sources if isinstance(s, dict) and s.get("source")})}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = topic.strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = make_model()
    start = time.monotonic()
    with open_sandbox() as backend:  # the sandbox is always stopped and removed, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        config = {"recursion_limit": RECURSION_LIMIT, "callbacks": [ToolTrace()]}
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]}, config=config)
            for _ in range(FAMILY_FIX_ROUNDS):  # RUBRIC 2.2: checked in code, not left to the LLM's diligence
                missing = missing_families(backend)
                if not missing:
                    break
                print(f"[research] fewer than {MIN_FAMILIES} source families; asking the lead to add {missing}",
                      file=sys.stderr)
                result = agent.invoke({"messages": result["messages"] + [
                    {"role": "user", "content": build_family_fix_prompt(missing)}]}, config=config)
            report_path = save_outputs(backend, topic, result["messages"], time.monotonic() - start,
                                       _model_name(model))
        except Exception as exc:  # noqa: BLE001 - any failed run: exit 1 and leave no report behind
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))

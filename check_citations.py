"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    if not sources:
        return ["no sources in sources.json"]
    problems = []
    by_n = {}
    seen_urls = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry is not an object: {entry!r}")
            continue
        n, url = entry.get("n"), entry.get("url")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source n={n!r} is not an integer")
            continue
        if n in by_n:
            problems.append(f"source number [{n}] appears twice in sources.json")
        by_n[n] = entry
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}] url is not http(s): {url!r}")
        elif url in seen_urls:
            problems.append(f"source [{n}] has the same url as source [{seen_urls[url]}]: {url}")
        else:
            seen_urls[url] = n

    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        problems.append("report has no '## References' heading")
        body, references = report_text, ""
    else:
        body, references = report_text[:headings[-1].start()], report_text[headings[-1].end():]

    cited = _cited_numbers(body)
    for n in sorted(cited - set(by_n)):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(set(by_n) - cited):
        problems.append(f"source [{n}] never cited")

    if headings:
        ref_lines = {}
        for match in _REF_LINE.finditer(references):
            n, rest = int(match.group(1)), match.group(2)
            if n in ref_lines:
                problems.append(f"reference [{n}] appears more than once")
                continue
            ref_lines[n] = rest
            if n not in by_n:
                problems.append(f"reference [{n}] is not a source in sources.json")
                continue
            urls = _URL.findall(rest)
            if len(urls) != 1:
                problems.append(f"reference [{n}] must hold exactly one URL, found {len(urls)}")
            elif urls[0].rstrip(".,;)") != by_n[n].get("url"):
                problems.append(f"reference [{n}] url {urls[0]} differs from sources.json {by_n[n].get('url')}")
        for n in sorted(set(by_n) - set(ref_lines)):
            problems.append(f"source [{n}] has no line in ## References")
    return problems


_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"(?m)^[ \t]*(?:[-*][ \t]+)?\[(\d+)\](.*)$")
_URL = re.compile(r"https?://[^\s<>\]]+")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")  # [3]  [1, 2]  [1-3]; not a Markdown link [3](url)


def _cited_numbers(body):
    """Numbers cited in the body; grouped forms [1, 2] / [1-3] are expanded, code spans are ignored."""
    cited = set()
    for i, segment in enumerate(_CODE.split(body)):
        if i % 2:  # odd segments are code
            continue
        for match in _GROUP.finditer(segment):
            for part in re.split(r"\s*,\s*", match.group(1)):
                span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
                if span:
                    a, b = int(span.group(1)), int(span.group(2))
                    cited.update(range(a, b + 1) if 0 <= b - a <= 200 else (a, b))
                else:
                    cited.add(int(part))
    return cited


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

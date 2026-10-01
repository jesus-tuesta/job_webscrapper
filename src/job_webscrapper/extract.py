from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "big-pickle")

PROMPT = """You extract job postings from a rendered careers webpage.

You are given the page URL, page title, a numbered list of candidate content BLOCKS
(each is a visible DOM chunk of the page), a numbered list of LINKS (anchor text +
href), and the page's plain TEXT.

HOW TO READ THE BLOCKS
Blocks are overlapping snapshots of the DOM, ordered shallowest-first, so the SAME
posting often appears in several blocks with different amounts of surrounding
surround. The listing you want is the block whose text contains the job title and,
ideally, "View role"/"Apply" language and the team/location lines around it. Ignore
blocks that are only navigation, promo tiles or footer copy.

YOUR TASK
Identify the individual job postings advertised on this page.

A job posting is a specific role that someone could apply to: it has (or implies) a job
title, and typically a location and/or team/division. Examples: "Software Engineer -
Platform (London)", "2026 Graduate Programme - Technology".

DO NOT treat these as job postings:
- Navigation, headers, footers, cookie banners, "About us", "Life at X", culture pages
- Department/category landing tiles, marketing cards, news items, blog posts
- Filter chips ("London", "Full-time"), pagination text ("Next", "Page 2"), result counts
- Search boxes, sign-in prompts, "No results found" placeholders

RULES
1. Use ONLY text present in the BLOCKS/LINKS. Never invent titles, URLs or details.
2. If a block looks like a job posting, try to match it to a LINK whose anchor text or
   href plausibly corresponds to that role. Only emit `url` if you can actually point at
   a LINK id from the provided list. Otherwise leave `url` as null.
3. `location`, `team`, `posted` must be copied verbatim from the page. Use null if absent.
4. If the page shows NO job postings, return an empty `jobs` array. That is a valid and
   useful answer -- do not pad the result with chrome.
5. If postings are truncated by pagination, set `truncated` to true and note it in
   `notes`. Do not guess at roles not visible.
6. Deduplicate: the same role often appears as both a BLOCK and a LINK. One entry per role.

Return ONLY a JSON object, no markdown fences, no commentary:
{{
  "jobs": [
    {{
      "title": "exact job title",
      "location": "verbatim location or null",
      "team": "verbatim team/department/category or null",
      "posted": "verbatim posting date or null",
      "url": "href from LINKS or null",
      "block_ref": "block id the role came from, e.g. B7"
    }}
  ],
  "total_visible": <int or null>,
  "truncated": <bool>,
  "notes": "<one short sentence, or null>"
}}

PAGE URL: {url}
PAGE TITLE: {title}

BLOCKS
{blocks}

LINKS
{links}
"""


class ExtractionError(RuntimeError):
    pass


def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # grab the outermost JSON object
    start = text.find("{")
    if start > 0:
        text = text[start:]
    return text.strip()


def build_payload(page, max_blocks: int = 120, max_links: int = 150, max_chars: int = 60_000) -> str:
    lines = []
    used = 0
    for i, b in enumerate(page.blocks[:max_blocks], 1):
        t = re.sub(r"\s+", " ", b["text"]).strip()
        line = f"B{i} [{b['tag']}] {t}\n"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    blocks = "".join(lines) or "(none)"

    llines = []
    used = 0
    n = 0
    for l in page.links:
        t = re.sub(r"\s+", " ", l.get("text") or "").strip()
        href = l.get("href") or ""
        if not t or not href:
            continue
        line = f"L{n} {t} -> {href}\n"
        if used + len(line) > max_chars // 2:
            break
        llines.append(line)
        used += len(line)
        n += 1
        if n >= max_links:
            break
    links = "".join(llines) or "(none)"
    return blocks, links


def call_llm(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 150) -> str:
    binary = shutil.which("opencode")
    if not binary:
        raise ExtractionError("`opencode` CLI not found on PATH")
    with tempfile.TemporaryDirectory() as td:
        pf = Path(td) / "page.txt"
        pf.write_text(prompt, encoding="utf-8")
        proc = subprocess.run(
            [binary, "run", "--format", "json", "--auto",
             "-m", f"opencode/{model}",
             "Extract the job postings from the attached page dump. Output only the JSON object.",
             "-f", str(pf)],
            capture_output=True, text=True, timeout=timeout,
        )
    if proc.returncode != 0:
        raise ExtractionError(f"opencode run failed ({proc.returncode}): {proc.stderr[-500:]}")
    out = []
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "text":
            out.append(ev["part"]["text"])
    if not out:
        raise ExtractionError(f"no text in opencode output: {proc.stdout[-500:]}")
    return "".join(out)


def extract_jobs(page, model: str = DEFAULT_MODEL, **kw) -> dict:
    blocks, links = build_payload(page, **kw)
    prompt = PROMPT.format(url=page.final_url or page.url, title=page.title, blocks=blocks, links=links)
    raw = call_llm(prompt, model=model)
    try:
        data = json.loads(_clean_json(raw))
    except json.JSONDecodeError as e:
        raise ExtractionError(f"LLM returned non-JSON: {e}\n---\n{raw[:800]}")
    if not isinstance(data, dict) or "jobs" not in data:
        raise ExtractionError(f"unexpected LLM shape: {list(data)[:10] if isinstance(data, dict) else type(data)}")
    return data
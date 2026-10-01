"""Tests for the headless fetch -> AI extraction -> polars pipeline.

Run with:  uv run pytest -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.job_webscrapper.extract import _clean_json, build_payload, extract_jobs
from src.job_webscrapper.fetch import Page
from src.job_webscrapper.pipeline import build_from_cache, load_companies, rows_to_frame
from src.job_webscrapper.report import _classify, failures


# --------------------------------------------------------------------------
# main.py CLI wiring
# --------------------------------------------------------------------------
def test_main_exposes_all_three_commands():
    from main import COMMANDS, build_parser

    assert set(COMMANDS) == {"scrape", "report", "run"}
    for cmd in COMMANDS:
        args = build_parser().parse_args([cmd] + (["--from-cache"] if cmd != "report" else []))
        assert args.command == cmd


def test_main_scrape_from_cache_is_offline(tmp_path: Path):
    from main import COMMANDS, build_parser

    (tmp_path / "acme.json").write_text(json.dumps({
        "company": "Acme", "source_url": "https://acme.test", "final_url": "https://acme.test",
        "page_title": "Jobs", "fetch_status": 200, "fetch_error": None, "truncated": False,
        "notes": None, "jobs": [{"title": "Quant Intern", "location": "London",
                                 "team": None, "posted": None, "url": None}],
    }))
    out = tmp_path / "j.parquet"
    args = build_parser().parse_args(
        ["scrape", "--from-cache", "--cache", str(tmp_path), "--out", str(out)]
    )
    df = COMMANDS["scrape"](args)
    assert df.height == 1 and df["company"].to_list() == ["Acme"]
    assert pl.read_parquet(out).height == 1


# --------------------------------------------------------------------------
# main.py CLI wiring
# --------------------------------------------------------------------------
def test_main_exposes_all_three_commands():
    from main import COMMANDS, build_parser

    assert set(COMMANDS) == {"scrape", "report", "run"}
    for cmd in COMMANDS:
        args = build_parser().parse_args([cmd] + (["--from-cache"] if cmd != "report" else []))
        assert args.command == cmd


def test_main_scrape_from_cache_is_offline(tmp_path: Path):
    from main import COMMANDS, build_parser

    (tmp_path / "acme.json").write_text(json.dumps({
        "company": "Acme", "source_url": "https://acme.test", "final_url": "https://acme.test",
        "page_title": "Jobs", "fetch_status": 200, "fetch_error": None, "truncated": False,
        "notes": None, "jobs": [{"title": "Quant Intern", "location": "London",
                                 "team": None, "posted": None, "url": None}],
    }))
    out = tmp_path / "j.parquet"
    args = build_parser().parse_args(
        ["scrape", "--from-cache", "--cache", str(tmp_path), "--out", str(out)]
    )
    df = COMMANDS["scrape"](args)
    assert df.height == 1 and df["company"].to_list() == ["Acme"]
    assert pl.read_parquet(out).height == 1


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
MOCK_BLOCKS = [
    {"tag": "li", "cls": "css-1q2dra3", "depth": 8, "has_more_siblings": True,
     "href": "", "text": "Quantitative Research Intern London Posted 12/09/2026 View role"},
    {"tag": "li", "cls": "css-1q2dra3", "depth": 8, "has_more_siblings": True,
     "href": "", "text": "Reliability Engineer London Posted 05/09/2026 View role"},
    {"tag": "div", "cls": "footer", "depth": 3, "has_more_siblings": False,
     "href": "", "text": "Cookie preferences. Privacy policy. Terms of use."},
]

MOCK_LINKS = [
    {"href": "https://example.test/en-GB/job/London/Quant-Intern_J1", "text": "Quantitative Research Intern"},
    {"href": "https://example.test/en-GB/job/London/Reliability-Engineer_J2", "text": "Reliability Engineer"},
    {"href": "https://example.test/about", "text": "About us"},
]


@pytest.fixture
def mock_page() -> Page:
    return Page(
        url="https://example.test/careers",
        final_url="https://example.test/careers",
        status=200,
        title="Search for Jobs",
        blocks=MOCK_BLOCKS,
        links=MOCK_LINKS,
        text="Quantitative Research Intern\nLondon\nReliability Engineer\nLondon",
    )


# --------------------------------------------------------------------------
# payload construction (offline)
# --------------------------------------------------------------------------
def test_build_payload_preserves_blocks_and_links(mock_page):
    blocks, links = build_payload(mock_page)
    assert "B1" in blocks and "B2" in blocks
    assert "Quantitative Research Intern" in blocks
    assert "L0" in links and "J1" in links
    assert links.count("\n") == 3


def test_build_payload_respects_limits(mock_page):
    blocks, links = build_payload(mock_page, max_blocks=1, max_links=1)
    assert "B1" in blocks and "B2" not in blocks
    assert "L0" in links and "L1" not in links


def test_job_links_filters_by_href_pattern(mock_page):
    found = {l["text"] for l in mock_page.job_links()}
    assert found == {"Quantitative Research Intern", "Reliability Engineer"}
    assert "About us" not in found  # no job-ish path segment


def test_clean_json_strips_fences_and_prose():
    assert json.loads(_clean_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(_clean_json('Here you go:\n{"a": 1}')) == {"a": 1}


# --------------------------------------------------------------------------
# dataframe assembly (offline)
# --------------------------------------------------------------------------
def test_rows_to_frame_empty_has_schema():
    df = rows_to_frame([])
    assert df.height == 0
    assert "title" in df.columns and "truncated" in df.columns


def test_rows_to_frame_normalises_nulls():
    df = rows_to_frame([{
        "company": "X", "title": " Analyst ", "location": "", "team": None,
        "posted": None, "url": None, "source_url": "u", "page_title": "t",
        "truncated": False, "extracted_at": "now", "fetch_status": 200, "fetch_error": None,
    }])
    row = df.row(0, named=True)
    assert row["title"] == "Analyst"
    assert row["location"] is None and row["url"] is None


def test_build_from_cache_round_trip(tmp_path: Path):
    (tmp_path / "acme.json").write_text(json.dumps({
        "company": "Acme", "source_url": "https://acme.test/jobs", "final_url": "https://acme.test/jobs",
        "page_title": "Jobs", "fetch_status": 200, "fetch_error": None,
        "truncated": False, "notes": None,
        "jobs": [
            {"title": "Quant Intern", "location": "London", "team": None,
             "posted": "12/09/2026", "url": "https://acme.test/job/1"},
            {"title": "SRE", "location": "London", "team": "Eng",
             "posted": None, "url": None},
        ],
    }))
    df = build_from_cache(tmp_path)
    assert df.height == 2
    assert df["company"].unique().to_list() == ["Acme"]
    assert df.filter(pl.col("title") == "SRE").row(0, named=True)["url"] is None


def test_build_from_cache_drops_empty_titles(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({
        "company": "A", "source_url": "u", "final_url": "u", "page_title": "t",
        "fetch_status": 200, "fetch_error": None, "truncated": False, "notes": None,
        "jobs": [{"title": "   ", "location": None, "team": None, "posted": None, "url": None}],
    }))
    assert build_from_cache(tmp_path).height == 0


# --------------------------------------------------------------------------
# status classification
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row,expected", [
    ({"http_status": 200, "n_jobs": 5, "has_error": False}, "ok"),
    ({"http_status": 200, "n_jobs": 0, "has_error": False}, "no_roles_found"),
    ({"http_status": 404, "n_jobs": 0, "has_error": False}, "http_404"),
    ({"http_status": 0, "n_jobs": 0, "has_error": True}, "fetch_error"),
])
def test_classify(row, expected):
    assert _classify(row) == expected


def test_failures_reads_cache(tmp_path: Path):
    (tmp_path / "b.json").write_text(json.dumps({
        "company": "B", "source_url": "u", "final_url": "u", "page_title": "t",
        "fetch_status": 500, "fetch_error": "boom", "truncated": False, "jobs": [],
    }))
    df = failures(tmp_path)
    assert df.row(0, named=True)["status"] == "fetch_error"


# --------------------------------------------------------------------------
# excel input
# --------------------------------------------------------------------------
def test_load_companies_filters_and_dedupes(tmp_path: Path, monkeypatch):
    xlsx = tmp_path / "wb.xlsx"
    pl.DataFrame({
        "Company": ["Acme", "Acme", "", "Globex", "0"],
        "Link": ["https://a.test", "https://a.test", "https://b.test", "https://g.test", "https://z.test"],
    }).write_excel(xlsx)
    df = load_companies(str(xlsx))
    assert df.height == 2
    assert set(df["company"].to_list()) == {"Acme", "Globex"}
    assert load_companies(str(xlsx), only=["globex"])["company"].to_list() == ["Globex"]


# --------------------------------------------------------------------------
# live: opencode reachability via its agentic CLI
# --------------------------------------------------------------------------
def test_opencode_cli_is_reachable():
    """The AI backend is the opencode CLI in agentic (non-interactive) mode."""
    import shutil
    import subprocess

    binary = shutil.which("opencode")
    assert binary, "opencode CLI not found on PATH"

    proc = subprocess.run(
        [binary, "run", "--format", "json", "Reply with exactly: PONG"],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-500:]

    events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    texts = [e["part"]["text"] for e in events if e.get("type") == "text"]
    assert texts, f"no text events in agentic output: {proc.stdout[-500:]}"
    assert "PONG" in "".join(texts)


@pytest.mark.network
def test_extract_jobs_end_to_end():
    """Live: render a real careers page and let the AI return structured roles."""
    from src.job_webscrapper.fetch import fetch

    url = "https://fil.wd3.myworkdayjobs.com/en-US/001?locationCountry=29247e57dbaf46fb855b224e03170bc7"
    page = fetch(url)
    assert page.ok, f"fetch failed: {page.error}"

    result = extract_jobs(page)
    assert isinstance(result.get("jobs"), list)
    assert result["jobs"], "expected at least one role"
    for job in result["jobs"]:
        assert job["title"]
        if job.get("url"):
            assert job["url"].startswith("http")
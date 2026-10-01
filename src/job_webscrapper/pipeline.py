from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import polars as pl

from .extract import ExtractionError, extract_jobs
from .fetch import close_browser, fetch, recycle_browser

DEFAULT_XLSX = "JobWebpages.xlsx"
DEFAULT_CACHE = Path(".cache")


def load_companies(xlsx: str, limit: int | None = None, only: list[str] | None = None) -> pl.DataFrame:
    with warnings.catch_warnings():
        # polars 2.0 will change read_excel's return type; irrelevant here.
        warnings.simplefilter("ignore", FutureWarning)
        raw = pl.read_excel(xlsx)
    df = raw.to_frame() if hasattr(raw, "to_frame") else raw
    df = df.select(
        pl.col("Company").cast(pl.Utf8).str.strip_chars().alias("company"),
        pl.col("Link").cast(pl.Utf8).str.strip_chars().alias("url"),
    ).filter(
        pl.col("company").is_not_null() & (pl.col("company") != "") &
        ~pl.col("company").str.contains(r"^0+(\.0+)?$") &
        pl.col("url").is_not_null() & (pl.col("url") != "") &
        pl.col("url").str.starts_with("http")
    ).unique(subset=["url"], keep="first")
    if only:
        pat = [re.compile(p, re.I) for p in only]
        df = df.filter(pl.col("company").str.to_lowercase().map_elements(
            lambda c: any(p.search(c) for p in pat), return_dtype=pl.Boolean
        ))
    if limit:
        df = df.head(limit)
    return df


def _norm(v) -> str | None:
    """Empty/whitespace-only strings become None so they don't pollute the frame."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={
            "company": pl.Utf8, "title": pl.Utf8, "location": pl.Utf8,
            "team": pl.Utf8, "posted": pl.Utf8, "url": pl.Utf8,
            "source_url": pl.Utf8, "page_title": pl.Utf8, "truncated": pl.Boolean,
            "extracted_at": pl.Utf8, "fetch_status": pl.Int32, "fetch_error": pl.Utf8,
        })
    return pl.DataFrame(
        [{k: (_norm(v) if isinstance(v, str) else v) for k, v in r.items()} for r in rows]
    )


def payload_rows(payload: dict, extracted_at: str) -> list[dict]:
    """Flatten one cached payload dict into job rows."""
    return [
        {
            "company": payload["company"],
            "title": (j.get("title") or "").strip(),
            "location": j.get("location") or None,
            "team": j.get("team") or None,
            "posted": j.get("posted") or None,
            "url": j.get("url") or None,
            "source_url": payload["source_url"],
            "page_title": payload.get("page_title"),
            "truncated": bool(payload.get("truncated")),
            "extracted_at": extracted_at,
            "fetch_status": payload.get("fetch_status") or 0,
            "fetch_error": payload.get("fetch_error"),
        }
        for j in (payload.get("jobs") or [])
    ]


def cache_key(company: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-")


def run(
    df: pl.DataFrame,
    cache_dir: Path = DEFAULT_CACHE,
    model: str | None = None,
    retries: int = 1,
    on_result=None,
) -> pl.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731

    for i, rec in enumerate(df.iter_rows(named=True), 1):
        company, url = rec["company"], rec["url"]
        cp = cache_dir / f"{cache_key(company)}.json"

        if cp.exists():
            payload = json.loads(cp.read_text())
            status = "cached"
            print(f"[{i}/{len(df)}] {company}: cached ({len(payload.get('jobs', []))} jobs)", file=sys.stderr)
        else:
            print(f"[{i}/{len(df)}] {company}: fetching {url[:70]}", file=sys.stderr)
            try:
                page = fetch(url)
            except Exception as e:  # noqa: BLE001
                print(f"    fetch crashed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                recycle_browser()  # drop the poisoned browser
                page = fetch(url)
            payload = {
                "company": company,
                "source_url": url,
                "final_url": page.final_url,
                "page_title": page.title,
                "fetch_status": page.status,
                "fetch_error": page.error,
                "truncated": False,
                "notes": None,
                "jobs": [],
            }
            if page.ok:
                for attempt in range(retries + 1):
                    try:
                        res = extract_jobs(page, model=model) if model else extract_jobs(page)
                        payload["jobs"] = res.get("jobs") or []
                        payload["truncated"] = bool(res.get("truncated"))
                        payload["total_visible"] = res.get("total_visible")
                        payload["notes"] = res.get("notes")
                        break
                    except ExtractionError as e:
                        if attempt == retries:
                            payload["fetch_error"] = f"extract_error: {e}"[:500]
                            print(f"    extract failed: {str(e)[:120]}", file=sys.stderr)
                        else:
                            time.sleep(5)
            else:
                print(f"    fetch failed: {page.error or page.status}", file=sys.stderr)
            cp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            status = "done"
            print(f"    -> {len(payload['jobs'])} jobs"
                  + (" (truncated)" if payload["truncated"] else ""), file=sys.stderr)

        rows.extend(payload_rows(payload, now()))
        if on_result:
            on_result(rows_to_frame(rows))

    return rows_to_frame(rows)


def build_from_cache(cache_dir: Path = DEFAULT_CACHE, out: str | None = None) -> pl.DataFrame:
    """Rebuild the jobs frame purely from the on-disk cache (no network, no LLM)."""
    rows: list[dict] = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for p in sorted(cache_dir.glob("*.json")):
        rows.extend(payload_rows(json.loads(p.read_text()), now))
    df = rows_to_frame(rows)
    if df.height:
        df = df.filter(pl.col("title") != "")
    if out:
        df.write_parquet(out)
    return df


def scrape(
    xlsx: str = DEFAULT_XLSX,
    cache_dir: Path = DEFAULT_CACHE,
    out: str = "jobs.parquet",
    limit: int | None = None,
    only: list[str] | None = None,
    model: str | None = None,
    retries: int = 1,
    append: bool = False,
    from_cache: bool = False,
) -> pl.DataFrame:
    """Run the full pipeline and write the resulting frame to *out*."""
    if from_cache:
        df = build_from_cache(cache_dir, out=out)
        close_browser()
        print(f"wrote {df.height} rows from cache -> {out}")
        return df

    companies = load_companies(xlsx, limit=limit, only=only)
    print(f"{len(companies)} companies to process")

    t0 = time.time()
    df = run(companies, cache_dir=cache_dir, model=model, retries=retries)
    close_browser()

    if append and Path(out).exists():
        df = pl.concat([pl.read_parquet(out), df], how="diagonal_relaxed")
        df = df.unique(subset=["company", "title", "url"], keep="first")

    df.write_parquet(out)
    print(f"wrote {len(df)} rows -> {out} in {time.time() - t0:.0f}s")
    return df
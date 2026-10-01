from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .pipeline import load_companies


def _classify(row: dict) -> str:
    if row["has_error"]:
        return "fetch_error"
    if row["http_status"] == 0:
        return "fetch_error"
    if 400 <= row["http_status"] < 500:
        return f"http_{row['http_status']}"
    if row["n_jobs"] == 0:
        return "no_roles_found"
    return "ok"


def failures(cache_dir: str = ".cache") -> pl.DataFrame:
    """Per-company outcome for every attempted fetch, including failures."""
    rows = []
    for p in sorted(Path(cache_dir).glob("*.json")):
        c = json.loads(p.read_text())
        err = c.get("fetch_error")
        rows.append({
            "company": c["company"],
            "source_url": c["source_url"],
            "final_url": c.get("final_url"),
            "http_status": c.get("fetch_status") or 0,
            "n_jobs": len(c.get("jobs") or []),
            "truncated": bool(c.get("truncated")),
            "has_error": bool(err),
            "error": (err or "")[:160] or None,
        })
    return pl.DataFrame(rows).with_columns(
        pl.struct(["http_status", "n_jobs", "has_error"])
        .map_elements(_classify, return_dtype=pl.Utf8)
        .alias("status")
    )


def coverage(cache_dir: str = ".cache", xlsx: str = "JobWebpages.xlsx") -> pl.DataFrame:
    """Companies-per-status counts across everything in the cache."""
    cache = [json.loads(p.read_text()) for p in Path(cache_dir).glob("*.json")]
    return (
        pl.DataFrame(
            {
                "company": [c["company"] for c in cache],
                "n_jobs": [len(c.get("jobs") or []) for c in cache],
                "http_status": [c.get("fetch_status") or 0 for c in cache],
                "has_error": [bool(c.get("fetch_error")) for c in cache],
            }
        )
        .with_columns(
            pl.struct(["http_status", "n_jobs", "has_error"])
            .map_elements(_classify, return_dtype=pl.Utf8)
            .alias("status")
        )
        .group_by("status")
        .agg(pl.len().alias("companies"))
        .sort("companies", descending=True)
    )


def summary(jobs: pl.DataFrame, cache_dir: str = ".cache", xlsx: str = "JobWebpages.xlsx") -> dict:
    """Headline counts plus the frames the CLI prints."""
    return {
        "companies_in_list": len(load_companies(xlsx)),
        "attempted": len(list(Path(cache_dir).glob("*.json"))),
        "job_rows": len(jobs),
        "companies_with_roles": jobs["company"].n_unique() if jobs.height else 0,
        "roles_missing_url": jobs.filter(pl.col("url").is_null()).height if jobs.height else 0,
        "coverage": coverage(cache_dir, xlsx),
        "failures": failures(cache_dir),
        "top_volume": (
            jobs.group_by("company")
            .agg(pl.len().alias("n"), pl.col("truncated").any().alias("trunc"))
            .sort("n", descending=True)
            .head(12)
            if jobs.height
            else pl.DataFrame()
        ),
    }


def report(jobs_path: str = "jobs.parquet", cache_dir: str = ".cache",
           xlsx: str = "JobWebpages.xlsx", csv: str | None = "jobs.csv") -> dict:
    """Load the jobs frame, build the coverage report, optionally export CSV."""
    jobs = pl.read_parquet(jobs_path)
    s = summary(jobs, cache_dir=cache_dir, xlsx=xlsx)

    print(f"companies in list : {s['companies_in_list']}")
    print(f"attempted         : {s['attempted']}")
    print(f"job rows          : {s['job_rows']}")
    print(f"companies w/ roles: {s['companies_with_roles']}")
    print(f"roles missing url : {s['roles_missing_url']}")
    print()
    print(s["coverage"])
    print()
    print("failures needing attention:")
    print(
        s["failures"].filter(pl.col("status") != "ok")
        .select("company", "status", "http_status", "error")
        .sort("status")
    )
    print()
    print("top by volume:")
    print(s["top_volume"])

    if csv:
        jobs.write_csv(csv)
        print(f"\ncsv -> {csv}")
    return s
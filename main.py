"""Entry point for the job webscrapper.

    python main.py scrape              # fetch + AI-extract -> jobs.parquet
    python main.py report              # coverage + failure breakdown
    python main.py run                 # scrape then report in one go

Run `python main.py <command> --help` for the options of each command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from src.job_webscrapper.pipeline import DEFAULT_CACHE, DEFAULT_XLSX, scrape  # noqa: E402
from src.job_webscrapper.report import report  # noqa: E402


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--xlsx", default=DEFAULT_XLSX, help="company/link spreadsheet")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="per-company result cache")
    ap.add_argument("--out", default="jobs.parquet", help="output parquet")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Scrape job roles from careers pages with a headless browser + AI extraction",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scrape", help="fetch each careers page and extract roles into polars")
    add_common(s)
    s.add_argument("--limit", type=int, default=None, help="only process the first N companies")
    s.add_argument("--only", nargs="*", default=None, help="regex filters on company name")
    s.add_argument("--model", default=None, help="override the opencode model")
    s.add_argument("--retries", type=int, default=1, help="AI extraction retries per page")
    s.add_argument("--append", action="store_true", help="union with the existing output file")
    s.add_argument("--from-cache", action="store_true",
                   help="skip network and AI work, rebuild the frame from the cache")

    r = sub.add_parser("report", help="summarise coverage and list failures")
    r.add_argument("--jobs", default="jobs.parquet")
    r.add_argument("--cache", default=str(DEFAULT_CACHE))
    r.add_argument("--xlsx", default=DEFAULT_XLSX)
    r.add_argument("--csv", default="jobs.csv")

    b = sub.add_parser("run", help="scrape then report")
    add_common(b)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--only", nargs="*", default=None)
    b.add_argument("--model", default=None)
    b.add_argument("--retries", type=int, default=1)
    b.add_argument("--append", action="store_true")
    b.add_argument("--from-cache", action="store_true")
    b.add_argument("--csv", default="jobs.csv")
    return ap


def cmd_scrape(args) -> pl.DataFrame:
    return scrape(
        xlsx=args.xlsx,
        cache_dir=Path(args.cache),
        out=args.out,
        limit=args.limit,
        only=args.only,
        model=args.model,
        retries=args.retries,
        append=args.append,
        from_cache=args.from_cache,
    )


def cmd_report(args) -> None:
    report(jobs_path=args.jobs, cache_dir=args.cache, xlsx=args.xlsx, csv=args.csv)


def cmd_run(args) -> None:
    jobs = cmd_scrape(args)
    print()
    report(jobs_path=args.out, cache_dir=args.cache, xlsx=args.xlsx, csv=args.csv)
    print()
    print("preview:")
    print(jobs.head(20))


COMMANDS = {"scrape": cmd_scrape, "report": cmd_report, "run": cmd_run}


def main() -> None:
    args = build_parser().parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
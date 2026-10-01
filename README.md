# job_webscrapper

Reads `JobWebpages.xlsx` (Company / careers URL), opens each URL in a **headless
Chromium** so the JS-heavy ATS widgets actually render, hands the reduced page to
**OpenCode in agentic mode** for extraction, and returns a tidy **polars** DataFrame
with one row per job role.

## Entry point

`main.py` is the single entry point, with three subcommands:

```bash
python main.py scrape      # fetch + AI-extract -> jobs.parquet
python main.py report      # coverage + failure breakdown -> jobs.csv
python main.py run         # scrape, then report, then preview
```

`--help` works on each. `uv run job-webscrapper <cmd>` maps to the same thing.

## How it works

```
JobWebpages.xlsx
   -> fetch.py    headless render -> DOM blocks + anchors + page text
   -> extract.py  opencode run --format json  -> strict JSON job schema
   -> pipeline.py assemble rows -> jobs.parquet
   -> report.py   coverage + failure breakdown -> jobs.csv
```

The extraction prompt (`extract.PROMPT`) tells the model what counts as a posting and
what is page chrome, requires verbatim `location` / `team` / `posted` values, and
forbids inventing URLs — `url` is only emitted when it can be traced to a link that
was actually on the page. Returning an empty `jobs` array is a valid answer, which is
what makes "0 roles found" meaningful rather than noise.

## Usage

```bash
python main.py scrape                              # full run
python main.py scrape --only Jane --limit 5       # targeted
python main.py scrape --from-cache                # rebuild frame, no network
python main.py run --append                        # scrape then report
```

Results are cached per company in `.cache/<company>.json`, so re-runs are cheap and
interrupted runs resume where they stopped.

## Output schema

`jobs.parquet`, one row per role:

| column | meaning |
| --- | --- |
| `company` | as listed in the spreadsheet |
| `title` | job title, verbatim |
| `location` / `team` / `posted` | verbatim from the page, null when absent |
| `url` | role URL, only when traceable to a link on the page |
| `source_url` | the careers page it came from |
| `truncated` | page was paginated, more roles exist |
| `fetch_status` / `fetch_error` | HTTP status or the reason the page failed |

## Tests

```bash
uv run pytest -m "not network"   # offline unit tests
uv run pytest -k opencode        # proves the AI backend is reachable
```

`test_opencode_cli_is_reachable` drives the same `opencode run --format json` path the
pipeline uses, so it covers the agentic integration end to end.

## Known limitations

- Most ATS boards paginate, so `truncated=true` rows are a partial view of that company.
- 403s (BNP, Patrizia) are bot blocks that headless Chromium does not get past.
- Some `no_roles_found` results are genuinely empty pages, others are landing pages
  that link out to a deeper search — those need a follow-up pass with per-site selectors.
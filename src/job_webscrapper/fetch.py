from __future__ import annotations

import re
from dataclasses import dataclass, field

from playwright.sync_api import sync_playwright

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
)

_BLOCK_TAGS = (
    "li, tr, article, [role=listitem], [data-job], [class*=job], [class*=Job], "
    "[class*=result], [class*=Result], [class*=card], [class*=Card], "
    "[class*=posting], [class*=vacanc], [class*=opportunit]"
)

_DROP_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "nav",
    "header",
    "footer",
    "form[role=search]",
    ".cookie",
    "[id*=cookie]",
    "[class*=cookie-banner]",
    "[class*=consent]",
    "[id*=onetrust]",
    "[aria-hidden=true]",
)

_JS = r"""
(sel) => {
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    const t = (el.innerText || '').replace(/\s+/g, ' ').trim();
    if (!t) continue;
    let href = null;
    const a = el.matches('a[href]') ? el : el.querySelector('a[href]');
    if (a) href = a.href;
    if (!href) {
      try { href = new URL(location.href).href; } catch (e) {}
    }
    const parent = el.parentElement;
    out.push({
      tag: el.tagName.toLowerCase(),
      cls: (el.className && el.className.toString
            ? el.className.toString().slice(0, 120) : ''),
      depth: (() => { let d = 0, p = el; while ((p = p.parentElement)) d++; return d; })(),
      has_more_siblings: parent ? parent.children.length > 1 : false,
      href,
      text: t,
    });
  }
  return out;
}
"""


_HREF_JOB = re.compile(
    r"/(job|jobs|role|roles|vacanc|vacanc(y|ies)|opportunit|opening|position|posting|career|search|results?)[/?#]",
    re.I,
)


@dataclass
class Page:
    url: str
    final_url: str
    status: int
    title: str
    blocks: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status < 400 and bool(self.blocks or self.text)

    def job_links(self) -> list[dict]:
        out, seen = [], set()
        for l in self.links:
            href = l.get("href") or ""
            if not href or href in seen or href.rstrip("/").lower() == self.final_url.rstrip("/").lower():
                continue
            if not _HREF_JOB.search(href):
                continue
            if len(l.get("text", "")) < 3:
                continue
            seen.add(href)
            out.append(l)
        return out


_LINKS_JS = r"""
() => {
  const out = [];
  for (const a of document.querySelectorAll('a[href]')) {
    const t = (a.innerText || a.textContent || '').replace(/\s+/g, ' ').trim();
    out.push({ href: a.href, text: t.slice(0, 200) });
  }
  return out;
}
"""


def _classify_blocks(raw: list[dict], max_blocks: int = 250) -> list[dict]:
    """Keep leaf-ish blocks that look like list entries, de-duplicated."""
    blocks: list[dict] = []
    seen: set[str] = set()
    raw.sort(key=lambda b: b["depth"])
    for b in raw:
        text = b["text"]
        if not (25 <= len(text) <= 1200):
            continue
        # a leaf block: no child block with a longer text at greater depth
        key = re.sub(r"\W+", " ", text.lower()).strip()[:160]
        if key in seen:
            continue
        seen.add(key)
        blocks.append(b)
        if len(blocks) >= max_blocks:
            break
    return blocks


_BROWSER = None
_PLAYWRIGHT = None


def _get_browser():  # noqa: D401
    """Reuse a single chromium process across fetches (launching is ~1s)."""
    global _BROWSER, _PLAYWRIGHT
    if _BROWSER is None:
        _PLAYWRIGHT = sync_playwright().start()
        _BROWSER = _PLAYWRIGHT.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
    return _BROWSER


def close_browser() -> None:
    global _BROWSER, _PLAYWRIGHT
    try:
        if _BROWSER:
            _BROWSER.close()
    finally:
        _BROWSER = None
        if _PLAYWRIGHT:
            _PLAYWRIGHT.stop()
            _PLAYWRIGHT = None


def recycle_browser() -> None:
    """Force a fresh chromium after a crash so later fetches are not poisoned."""
    global _BROWSER
    try:
        if _BROWSER:
            _BROWSER.close()
    except Exception:
        pass
    _BROWSER = None


def fetch(
    url: str,
    *,
    timeout: int = 20_000,
    wait_ms: int = 2_500,
    settle_ms: int = 1_200,
    max_blocks: int = 250,
) -> Page:
    ctx = _get_browser().new_context(
        user_agent=UA,
        viewport={"width": 1440, "height": 1000},
        locale="en-GB",
    )
    page = ctx.new_page()
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        status = resp.status if resp else 0
        page.wait_for_timeout(wait_ms)

        # nudge lazy lists: scroll to bottom, wait for more to load
        try:
            page.evaluate(
                """async () => {
                    const h = document.body.scrollHeight;
                    for (let y = 0; y < h; y += 1200) {
                        window.scrollTo(0, y);
                        await new Promise(r => setTimeout(r, 60));
                    }
                    window.scrollTo(0, 0);
                }"""
            )
            page.wait_for_timeout(settle_ms)
        except Exception:
            pass

        for sel in _DROP_SELECTORS:
            try:
                page.eval_on_selector_all(
                    sel, "els => els.forEach(e => e.remove && e !== document.body && e.remove())"
                )
            except Exception:
                pass

        title = page.title()
        final_url = page.url
        text = page.evaluate("() => document.body ? document.body.innerText : ''")
        text = re.sub(r"\n{3,}", "\n\n", text)
        raw = page.evaluate(_JS, _BLOCK_TAGS)
        blocks = _classify_blocks(raw, max_blocks=max_blocks)
        links = page.evaluate(_LINKS_JS)
    except Exception as e:  # noqa: BLE001
        return Page(url=url, final_url=url, status=0, title="", error=f"{type(e).__name__}: {e}")
    finally:
        try:
            ctx.close()
        except Exception:
            pass

    return Page(
        url=url,
        final_url=final_url,
        status=status,
        title=title,
        blocks=blocks,
        links=links,
        text=text,
    )
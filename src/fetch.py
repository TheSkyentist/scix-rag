"""
Fetch raw content (HTML or PDF) for every paper that has neither a
converted Markdown file nor a raw file already waiting to be converted.
Pure fetch -- no conversion happens here, that's convert.py's job. HTML is
still validated by attempting a real conversion before it's kept (the
whole point of falling back to PDF is "when HTML doesn't convert to
enough text"), but the Markdown produced by that validation pass is
thrown away -- convert.py does the conversion that actually gets kept, so
there's exactly one place that owns "what does data/markdown/*.md contain".

Covers both:
  - arxiv_id IS NOT NULL: arXiv HTML5 (then ar5iv), falling back to the
    arXiv PDF if neither HTML source converts to enough text.
  - arxiv_id IS NULL: ADS's own hosted PDF sources (PUB_PDF, ADS_PDF,
    EPRINT_PDF, tried in that priority order via the public link-gateway --
    no upfront open-access-flag gating, since ADS's own *_OPENACCESS
    property flags are unreliable; the real HTTP response is ground
    truth). `esources` (which sources ADS claims to have) is looked up
    once per paper and cached in papers.duckdb -- real ADS metadata worth
    not re-fetching, not pipeline-status bookkeeping.

No status columns are written or read beyond `esources` -- "done" is
whatever's on disk: a paper is skipped if data/markdown/{stem}.md or
data/html/{stem}.html or data/pdf/{stem}.pdf already exists. This makes
every run trivially resumable, at the cost of retrying genuinely
unrecoverable papers on every run (no permanent "gave up" marker) -- a
deliberate simplicity-over-efficiency tradeoff.

Run directly:
    pixi run fetch
    pixi run fetch --limit 5   # smoke-test on a handful first
"""

from __future__ import annotations

import time

import duckdb
import httpx

import config

from .ads_query import _fetch_many_by_bibcode, _session
from .html_to_markdown import convert

_ARXIV_HTML_URLS = [
    "https://arxiv.org/html/{arxiv_id}",
    "https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
]
_GATEWAY = "https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/{esource}"
_ESOURCE_PRIORITY = ["PUB_PDF", "ADS_PDF", "EPRINT_PDF"]


def _done(stem: str) -> bool:
    return (
        (config.MARKDOWN_DIR / f"{stem}.md").exists()
        or (config.HTML_DIR / f"{stem}.html").exists()
        or (config.PDF_DIR / f"{stem}.pdf").exists()
    )


def _fetch_url_pdf(url: str, client: httpx.Client) -> bytes | None:
    try:
        resp = client.get(url, follow_redirects=True, timeout=60)
    except httpx.RequestError:
        return None
    if resp.status_code != 200 or "application/pdf" not in resp.headers.get("content-type", ""):
        return None
    return resp.content


def _fetch_arxiv_html(arxiv_id: str, client: httpx.Client) -> str | None:
    """Try arXiv-native then ar5iv HTML, validated by actually converting it."""
    for url_template in _ARXIV_HTML_URLS:
        url = url_template.format(arxiv_id=arxiv_id)
        try:
            resp = client.get(url, follow_redirects=True, timeout=30)
        except httpx.RequestError:
            continue
        if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
            continue
        try:
            if len(convert(resp.text)) >= config.MIN_MARKDOWN_CHARS:
                return resp.text
        except Exception:
            continue
    return None


def _fetch_arxiv(arxiv_id: str, client: httpx.Client) -> tuple[str, bytes | str] | None:
    html = _fetch_arxiv_html(arxiv_id, client)
    if html:
        return "html", html
    pdf = _fetch_url_pdf(f"https://arxiv.org/pdf/{arxiv_id}", client)
    if pdf:
        return "pdf", pdf
    return None


def _fetch_ads(bibcode: str, esources: list[str] | None, client: httpx.Client) -> tuple[str, bytes] | None:
    for esource in _ESOURCE_PRIORITY:
        if esource not in (esources or []):
            continue
        pdf = _fetch_url_pdf(_GATEWAY.format(bibcode=bibcode, esource=esource), client)
        if pdf:
            return "pdf", pdf
    return None


def _backfill_esources(con: duckdb.DuckDBPyConnection, bibcodes: list[str]) -> None:
    """One-time-per-paper: cache which full-text sources ADS claims to have,
    only for the non-arXiv papers in this run that don't have it yet."""
    if not bibcodes:
        return
    print(f"Looking up esources for {len(bibcodes)} non-arXiv papers...")
    with _session() as client:
        docs = _fetch_many_by_bibcode(client, bibcodes)
    rows = [(d.get("esources") or [], d["bibcode"]) for d in docs if d.get("bibcode")]
    if not rows:
        return  # ADS didn't resolve any of these bibcodes this time; retry next run
    con.executemany("UPDATE papers SET esources = ? WHERE bibcode = ?", rows)
    con.commit()


def fetch(limit: int | None) -> None:
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS esources VARCHAR[]")

    all_papers = con.execute("SELECT bibcode, arxiv_id FROM papers").fetchall()
    todo_bibcodes = [b for b, a in all_papers if not _done(a or b)]
    if limit:
        todo_bibcodes = todo_bibcodes[:limit]

    need_esources = [
        r[0] for r in con.execute(
            "SELECT bibcode FROM papers WHERE bibcode = ANY(?) AND arxiv_id IS NULL AND esources IS NULL",
            [todo_bibcodes],
        ).fetchall()
    ]
    _backfill_esources(con, need_esources)

    todo = con.execute(
        "SELECT bibcode, arxiv_id, esources FROM papers WHERE bibcode = ANY(?)",
        [todo_bibcodes],
    ).fetchall()
    print(f"Fetching {len(todo)} papers")

    with httpx.Client(headers={"User-Agent": "arXivRAG/0.1 (research pipeline)"}) as client:
        for bibcode, arxiv_id, esources in todo:
            stem = arxiv_id or bibcode
            if arxiv_id:
                result = _fetch_arxiv(arxiv_id, client)
                delay = config.ARXIV_DELAY
            else:
                result = _fetch_ads(bibcode, esources, client)
                delay = config.ADS_DELAY

            if result:
                kind, content = result
                if kind == "html":
                    (config.HTML_DIR / f"{stem}.html").write_text(content, encoding="utf-8")
                else:
                    (config.PDF_DIR / f"{stem}.pdf").write_bytes(content)
                print(f"  [{kind}] {stem}")
            else:
                print(f"  [fail] {stem}: nothing fetchable")
            time.sleep(delay)

    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch raw HTML/PDF for papers not yet converted")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending papers")
    args = parser.parse_args()

    fetch(limit=args.limit)

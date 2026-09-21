"""
Fetch arXiv HTML5 for papers in papers.duckdb and convert to clean Markdown.
If HTML conversion fails outright (LaTeXML fatal error, no HTML5 available),
falls back to fetching the raw PDF and letting Arcana's own Docling
pipeline handle it at upload time -- lower fidelity than our own cleaning,
but better than dropping the paper from the corpus entirely.

Status columns on `papers` track progress (added if missing, never reset
on re-run, same upsert-safe pattern as ads_query.py):
    html_status:    pending | fetched | no_html | no_arxiv_id
    pdf_status:     pending | fetched | failed | not_needed | no_arxiv_id
    md_status:      pending | converted | failed          (html path only)
    ingest_method:  pending | html | pdf | none            (which path won)

Run directly:
    pixi run fetch-convert
    pixi run fetch-convert --limit 5   # smoke-test on a handful first
"""

from __future__ import annotations

import time

import duckdb
import httpx

import config

from .html_to_markdown import convert

_ARXIV_HTML_URLS = [
    "https://arxiv.org/html/{arxiv_id}",
    "https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
]

# Below this, treat a conversion as empty/failed. Calibrated against the
# corpus: smallest genuine converted paper so far is ~1900 chars; ar5iv's
# own "conversion had a fatal error" placeholder still contains an (empty)
# <article class="ltx_document"> plus a stray error <div>, so a naive
# "ltx_document in html" check isn't enough -- validate by actually
# running the real converter and checking it produced substance.
_MIN_MARKDOWN_CHARS = 500


def _ensure_columns(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS html_status VARCHAR DEFAULT 'pending'")
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS md_status VARCHAR DEFAULT 'pending'")
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS pdf_status VARCHAR DEFAULT 'pending'")
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS ingest_method VARCHAR DEFAULT 'pending'")
    con.execute("""
        UPDATE papers SET html_status = 'no_arxiv_id', pdf_status = 'no_arxiv_id', ingest_method = 'none'
        WHERE arxiv_id IS NULL AND html_status = 'pending'
    """)


def _fetch_one(arxiv_id: str, client: httpx.Client) -> str | None:
    """Try arXiv-native then ar5iv HTML. Validated by actually converting it."""
    for url_template in _ARXIV_HTML_URLS:
        url = url_template.format(arxiv_id=arxiv_id)
        try:
            resp = client.get(url, follow_redirects=True, timeout=30)
        except httpx.RequestError:
            continue
        if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
            continue
        try:
            if len(convert(resp.text)) >= _MIN_MARKDOWN_CHARS:
                return resp.text
        except Exception:
            continue
    return None


def _fetch_pdf(arxiv_id: str, client: httpx.Client) -> bytes | None:
    """Fallback for papers whose HTML conversion failed outright."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        resp = client.get(url, follow_redirects=True, timeout=60)
    except httpx.RequestError:
        return None
    if resp.status_code != 200 or "application/pdf" not in resp.headers.get("content-type", ""):
        return None
    return resp.content


def fetch(limit: int | None, force: bool) -> None:
    """Fetch HTML for each paper; fall back to raw PDF if HTML conversion fails."""
    con = duckdb.connect(str(config.DB_PATH))
    _ensure_columns(con)

    where = "arxiv_id IS NOT NULL" if force else "html_status = 'pending'"
    q = f"SELECT bibcode, arxiv_id FROM papers WHERE {where}"
    if limit:
        q += f" LIMIT {limit}"
    todo = con.execute(q).fetchall()
    print(f"Fetching {len(todo)} papers (HTML, falling back to PDF)")

    with httpx.Client(headers={"User-Agent": "arXivRAG/0.1 (research pipeline)"}) as client:
        for bibcode, arxiv_id in todo:
            html = _fetch_one(arxiv_id, client)
            if html:
                (config.HTML_DIR / f"{arxiv_id}.html").write_text(html, encoding="utf-8")
                con.execute(
                    "UPDATE papers SET html_status='fetched', pdf_status='not_needed', ingest_method='html' WHERE bibcode=?",
                    [bibcode],
                )
                print(f"  [html] {arxiv_id}")
            else:
                pdf = _fetch_pdf(arxiv_id, client)
                if pdf:
                    (config.PDF_DIR / f"{arxiv_id}.pdf").write_bytes(pdf)
                    con.execute(
                        "UPDATE papers SET html_status='no_html', pdf_status='fetched', ingest_method='pdf' WHERE bibcode=?",
                        [bibcode],
                    )
                    print(f"  [pdf]  {arxiv_id} ({len(pdf)} bytes, HTML unavailable)")
                else:
                    con.execute(
                        "UPDATE papers SET html_status='no_html', pdf_status='failed', ingest_method='none' WHERE bibcode=?",
                        [bibcode],
                    )
                    print(f"  [fail] {arxiv_id}: no HTML or PDF available")
            time.sleep(config.ARXIV_DELAY)

    con.close()


def convert_markdown(limit: int | None, force: bool) -> None:
    con = duckdb.connect(str(config.DB_PATH))
    _ensure_columns(con)

    where = "html_status = 'fetched'" if force else "html_status = 'fetched' AND md_status = 'pending'"
    q = f"SELECT bibcode, arxiv_id FROM papers WHERE {where}"
    if limit:
        q += f" LIMIT {limit}"
    todo = con.execute(q).fetchall()
    print(f"Converting {len(todo)} papers to Markdown")

    for bibcode, arxiv_id in todo:
        html_path = config.HTML_DIR / f"{arxiv_id}.html"
        if not html_path.exists():
            print(f"  [missing html] {arxiv_id}")
            continue
        try:
            md = convert(html_path.read_text(encoding="utf-8"))
            if len(md) < _MIN_MARKDOWN_CHARS:
                status = "failed"
                print(f"  [fail] {arxiv_id}: converted to only {len(md)} chars, treating as empty")
            else:
                (config.MARKDOWN_DIR / f"{arxiv_id}.md").write_text(md, encoding="utf-8")
                status = "converted"
                print(f"  [ok]   {arxiv_id} ({len(md)} bytes)")
        except Exception as e:
            status = "failed"
            print(f"  [fail] {arxiv_id}: {e}")
        con.execute("UPDATE papers SET md_status = ? WHERE bibcode = ?", [status, bibcode])

    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch arXiv HTML and convert to clean Markdown")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending papers")
    parser.add_argument("--force", action="store_true", help="Re-fetch/re-convert regardless of status")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip the HTML fetch step (convert only)")
    parser.add_argument("--skip-convert", action="store_true", help="Skip the Markdown conversion step (fetch only)")
    args = parser.parse_args()

    if not args.skip_fetch:
        fetch(limit=args.limit, force=args.force)
    if not args.skip_convert:
        convert_markdown(limit=args.limit, force=args.force)

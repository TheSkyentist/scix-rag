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

Papers where every source answered definitively "no" are appended to
`data/fetch_ignore.txt` (gitignored; same format as convert_ignore.txt)
and skipped on later runs, so they aren't re-tried every time. Network
errors, timeouts, 429s and 5xx are *not* recorded -- those retry next run.
Delete a line (or pass --retry-failed) to try a paper again.

`--refresh` re-queries ADS for papers that still have no arxiv_id (a preprint
is often merged into the journal record after we ingested it, and ads_query
never revisits known bibcodes), updating arxiv_id / alternate_bibcode /
esources, then fetches as usual. A paper that gains an arxiv_id changes stem
(bibcode -> arxiv_id), so its old ignore line stops matching and it is retried
via arXiv; once the arXiv copy lands, leftover files under the old stem are
deleted, and `chunk` re-chunks it (it notices chunks.arxiv_id no longer matches).

Run directly:
    pixi run fetch
    pixi run fetch --limit 5   # smoke-test on a handful first
    pixi run fetch --retry-failed
    pixi run fetch --refresh --dry-run [--since 2008]   # report what would change
    pixi run fetch --refresh [--since 2008]
"""

from __future__ import annotations

import time

import duckdb
import httpx

import config

from .ads_query import _fetch_many_by_bibcode, _session
from .canonicalize import run as canonicalize
from .html_to_markdown import convert

_ARXIV_HTML_URLS = [
    "https://arxiv.org/html/{arxiv_id}",
    "https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
]
_GATEWAY = "https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/{esource}"
_ESOURCE_PRIORITY = ["PUB_PDF", "ADS_PDF", "EPRINT_PDF"]


class _Transient(Exception):
    """A failure that says nothing about whether the paper is fetchable."""


def _load_failed() -> set[str]:
    if not config.FETCH_IGNORE_PATH.exists():
        return set()
    lines = config.FETCH_IGNORE_PATH.read_text(encoding="utf-8").splitlines()
    return {name for line in lines if (name := line.split("#", 1)[0].strip())}


def _record_failed(stem: str, reason: str) -> None:
    new_file = not config.FETCH_IGNORE_PATH.exists()
    with config.FETCH_IGNORE_PATH.open("a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "# Papers `pixi run fetch` found nothing fetchable for (appended automatically;\n"
                "# one stem per line). Skipped unless: pixi run fetch --retry-failed\n"
            )
        f.write(f"{stem}  # {reason}\n")


def _check(resp: httpx.Response) -> None:
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _Transient(f"HTTP {resp.status_code}")


def _done(stem: str) -> bool:
    return (
        (config.MARKDOWN_DIR / f"{stem}.md").exists()
        or (config.HTML_DIR / f"{stem}.html").exists()
        or (config.PDF_DIR / f"{stem}.pdf").exists()
    )


def _fetch_url_pdf(url: str, client: httpx.Client) -> bytes | None:
    try:
        resp = client.get(url, follow_redirects=True, timeout=60)
    except httpx.RequestError as e:
        raise _Transient(type(e).__name__) from e
    _check(resp)
    if resp.status_code != 200 or "application/pdf" not in resp.headers.get("content-type", ""):
        return None
    return resp.content


def _fetch_arxiv_html(arxiv_id: str, client: httpx.Client) -> str | None:
    """Try arXiv-native then ar5iv HTML, validated by actually converting it."""
    for url_template in _ARXIV_HTML_URLS:
        url = url_template.format(arxiv_id=arxiv_id)
        try:
            resp = client.get(url, follow_redirects=True, timeout=30)
        except httpx.RequestError as e:
            raise _Transient(type(e).__name__) from e
        _check(resp)
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


def _refresh_links(con: duckdb.DuckDBPyConnection, since: int | None, dry_run: bool) -> None:
    """Re-query ADS for papers without an arxiv_id and let canonicalize fold in what changed
    (preprint merged into the journal record, twin rows, retired bibcodes)."""
    sql, params = "SELECT bibcode FROM papers WHERE arxiv_id IS NULL", []
    if since:
        sql, params = sql + " AND year >= ?", [since]
    bibcodes = [r[0] for r in con.execute(sql, params).fetchall()]
    print(f"Refreshing ADS links for {len(bibcodes)} papers without an arxiv_id...")
    with _session() as client:
        docs = _fetch_many_by_bibcode(client, bibcodes)
    canonicalize(con, docs, apply=not dry_run)


def _drop_old_stem(bibcode: str) -> None:
    """A paper that gained an arxiv_id switched stems; once its arXiv copy is on disk,
    delete what's left under the old bibcode stem (chunk.py re-chunks from the new one)."""
    for directory, ext in ((config.MARKDOWN_DIR, "md"), (config.HTML_DIR, "html"), (config.PDF_DIR, "pdf")):
        old = directory / f"{bibcode}.{ext}"
        if old.exists():
            old.unlink()
            print(f"  [drop] {old.relative_to(config.ROOT)}")


def fetch(
    limit: int | None,
    retry_failed: bool = False,
    refresh: bool = False,
    since: int | None = None,
    dry_run: bool = False,
) -> None:
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS esources VARCHAR[]")

    if refresh:
        _refresh_links(con, since, dry_run)
        if dry_run:
            con.close()
            return

    all_papers = con.execute("SELECT bibcode, arxiv_id FROM papers").fetchall()
    failed = set() if retry_failed else _load_failed()
    todo_bibcodes = [b for b, a in all_papers if not _done(a or b) and (a or b) not in failed]
    if failed:
        print(f"Skipping {len(failed)} papers listed in {config.FETCH_IGNORE_PATH.name} (--retry-failed to retry)")
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
            delay = config.ARXIV_DELAY if arxiv_id else config.ADS_DELAY
            try:
                if arxiv_id:
                    result = _fetch_arxiv(arxiv_id, client)
                else:
                    result = _fetch_ads(bibcode, esources, client)
            except _Transient as e:
                print(f"  [retry] {stem}: {e} (not recorded)")
                time.sleep(delay)
                continue

            if result:
                kind, content = result
                if kind == "html":
                    (config.HTML_DIR / f"{stem}.html").write_text(content, encoding="utf-8")
                else:
                    (config.PDF_DIR / f"{stem}.pdf").write_bytes(content)
                print(f"  [{kind}] {stem}")
                if arxiv_id:
                    _drop_old_stem(bibcode)
            else:
                print(f"  [fail] {stem}: nothing fetchable")
                if esources is None and not arxiv_id:
                    continue  # esources lookup failed upstream; not evidence the paper is unfetchable
                if arxiv_id:
                    reason = "no arXiv HTML/PDF"
                else:
                    reason = f"no ADS PDF source (esources: {', '.join(esources or []) or 'none'})"
                if stem not in failed:
                    _record_failed(stem, reason)
                    failed.add(stem)
            time.sleep(delay)

    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch raw HTML/PDF for papers not yet converted")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending papers")
    parser.add_argument("--retry-failed", action="store_true", help="Also retry papers listed in fetch_ignore.txt")
    parser.add_argument("--refresh", action="store_true", help="Re-query ADS for papers with no arxiv_id first")
    parser.add_argument("--since", type=int, default=None, help="With --refresh: only papers from this year on")
    parser.add_argument("--dry-run", action="store_true", help="With --refresh: report what would change, write nothing, don't fetch")
    args = parser.parse_args()
    if (args.since or args.dry_run) and not args.refresh:
        parser.error("--since/--dry-run need --refresh")

    fetch(limit=args.limit, retry_failed=args.retry_failed, refresh=args.refresh, since=args.since, dry_run=args.dry_run)

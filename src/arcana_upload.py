"""
Upload converted paper Markdown (or, for the PDF fallback, the raw PDF)
to Arcana, one file per paper.

Filename at upload time is the arxiv_id -- the stable join key back to
papers.duckdb. Arcana's own citations at query time only give back
"source filename + page", so the filename has to carry this key.

Run directly:
    pixi run arcana-upload --arxiv-id 2506.05459
"""

from __future__ import annotations

import time

import duckdb
import httpx

import config


def _check_config() -> None:
    if not config.ARCANA_TOKEN:
        raise RuntimeError("ARCANA_TOKEN/SAIA_API_KEY not set in .env")
    if not config.ARCANA_NAME:
        raise RuntimeError("ARCANA_NAME not set in .env")


def _resolve_upload(arxiv_id: str, ingest_method: str) -> tuple:
    if ingest_method == "pdf":
        path = config.PDF_DIR / f"{arxiv_id}.pdf"
        return path, f"{arxiv_id}.pdf", "application/pdf"
    path = config.MARKDOWN_DIR / f"{arxiv_id}.md"
    return path, f"{arxiv_id}.md", "text/markdown"


def _post_file(client: httpx.Client, path, filename: str, content_type: str) -> httpx.Response:
    content = path.read_bytes()
    if len(content) > config.ARCANA_MAX_FILE_BYTES:
        raise ValueError(
            f"{filename} is {len(content)} bytes, exceeds Arcana's documented "
            f"{config.ARCANA_MAX_FILE_BYTES} byte max -- not uploading"
        )
    url = f"{config.ARCANA_BASE_URL}/arcana/{config.ARCANA_NAME}/files/"
    resp = client.post(url, files={"file": (filename, content, content_type)})
    resp.raise_for_status()
    return resp


def upload_paper(arxiv_id: str, ingest_method: str = "html") -> None:
    """Upload one paper. Does NOT make it searchable -- see generate_index()."""
    _check_config()

    path, filename, content_type = _resolve_upload(arxiv_id, ingest_method)
    if not path.exists():
        raise FileNotFoundError(f"No {ingest_method} file for {arxiv_id} at {path}")

    print(f"Uploading {filename} ({path.stat().st_size} bytes) to arcana/{config.ARCANA_NAME}...")
    headers = {"Authorization": config.ARCANA_TOKEN, "accept": "application/json"}
    with httpx.Client(headers=headers, timeout=300) as client:
        resp = _post_file(client, path, filename, content_type)
    print(f"HTTP {resp.status_code}")


def generate_index() -> float:
    """
    Trigger indexing/embedding for the whole Arcana. This is a separate,
    blocking, arcana-wide call -- not per file. Uploading N files without
    calling this leaves them stored but not searchable. For batch uploads,
    call this once after all files are uploaded, not after each one.

    Returns elapsed seconds.
    """
    _check_config()
    url = f"{config.ARCANA_BASE_URL}/arcana/{config.ARCANA_NAME}/generate-index"
    print("Generating index (blocks until done)...")
    t0 = time.time()
    with httpx.Client(headers={"Authorization": config.ARCANA_TOKEN}, timeout=1800) as client:
        resp = client.post(url)
    elapsed = time.time() - t0
    print(f"HTTP {resp.status_code} in {elapsed:.1f}s: {resp.text}")
    resp.raise_for_status()
    return elapsed


def upload_all(force: bool = False) -> dict[str, float | int]:
    """Upload every ready-but-not-yet-uploaded paper (html or pdf path). Does not index."""
    _check_config()
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS arcana_status VARCHAR DEFAULT 'pending'")

    ready = "(ingest_method = 'html' AND md_status = 'converted') OR (ingest_method = 'pdf' AND pdf_status = 'fetched')"
    where = ready if force else f"({ready}) AND arcana_status = 'pending'"
    todo = con.execute(f"SELECT bibcode, arxiv_id, ingest_method FROM papers WHERE {where}").fetchall()
    print(f"Uploading {len(todo)} papers to arcana/{config.ARCANA_NAME}")

    headers = {"Authorization": config.ARCANA_TOKEN, "accept": "application/json"}

    ok = fail = 0
    t0 = time.time()
    with httpx.Client(headers=headers, timeout=300) as client:
        for bibcode, arxiv_id, ingest_method in todo:
            path, filename, content_type = _resolve_upload(arxiv_id, ingest_method)
            if not path.exists():
                print(f"  [missing] {arxiv_id}")
                con.execute("UPDATE papers SET arcana_status = 'failed' WHERE bibcode = ?", [bibcode])
                fail += 1
                continue
            try:
                _post_file(client, path, filename, content_type)
                status = "uploaded"
                ok += 1
                print(f"  [ok]   {arxiv_id} ({ingest_method})")
            except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
                status = "failed"
                fail += 1
                print(f"  [fail] {arxiv_id}: {e}")
            con.execute("UPDATE papers SET arcana_status = ? WHERE bibcode = ?", [status, bibcode])
    elapsed = time.time() - t0
    con.close()
    print(f"\nUpload phase done in {elapsed:.1f}s: {ok} ok, {fail} failed")
    return {"ok": ok, "fail": fail, "upload_seconds": elapsed}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload paper(s) to Arcana and (re)index")
    parser.add_argument("--arxiv-id", help="Upload this one paper's converted Markdown")
    parser.add_argument("--all", action="store_true", help="Upload every converted, not-yet-uploaded paper")
    parser.add_argument("--force", action="store_true", help="With --all, re-upload regardless of arcana_status")
    parser.add_argument("--no-index", action="store_true", help="Skip triggering generate-index after upload")
    args = parser.parse_args()

    if args.arxiv_id:
        con = duckdb.connect(str(config.DB_PATH))
        row = con.execute("SELECT ingest_method FROM papers WHERE arxiv_id = ?", [args.arxiv_id]).fetchone()
        con.close()
        upload_paper(args.arxiv_id, ingest_method=(row[0] if row else "html"))
    elif args.all:
        upload_all(force=args.force)
    if not args.no_index:
        generate_index()

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

    con = duckdb.connect(str(config.DB_PATH))
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS arcana_status VARCHAR DEFAULT 'pending'")
    con.execute(
        "UPDATE papers SET arcana_status = 'uploaded' WHERE arxiv_id = ?", [arxiv_id]
    )
    con.close()


def generate_index() -> float:
    """
    Trigger indexing/embedding for the whole Arcana. This is a separate,
    blocking, arcana-wide call -- not per file. Uploading N files without
    calling this leaves them stored but not searchable. For batch uploads,
    call this once after all files are uploaded, not after each one.

    Note: the POST reliably 504s client-side after GWDG's own ~60s gateway
    timeout on anything nontrivial, long before the job actually finishes
    server-side -- this raises on that. Use trigger_index_and_wait() for
    anything that needs the real outcome.

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


def get_index_info() -> dict:
    _check_config()
    with httpx.Client(headers={"Authorization": config.ARCANA_TOKEN}, timeout=30) as client:
        resp = client.get(f"{config.ARCANA_BASE_URL}/arcana/")
        resp.raise_for_status()
    return resp.json()[0]["index_info"] or {}


def wait_for_index(poll_interval: float = 60.0, max_wait: float = 6.5 * 3600) -> dict:
    """Poll until index_status leaves PENDING/INDEXING/RUNNING. Returns the final index_info."""
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed > max_wait:
            raise TimeoutError(f"Still polling after {elapsed:.0f}s with no terminal status")
        try:
            info = get_index_info()
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            print(f"  [poll {elapsed:6.0f}s] transient error, retrying: {e!r}")
            time.sleep(poll_interval)
            continue
        status = info.get("index_status")
        print(f"  [poll {elapsed:6.0f}s] status={status} "
              f"files_indexed={info.get('total_files_indexed')} chunks_indexed={info.get('total_chunks_indexed')}")
        if status not in ("PENDING", "INDEXING", "RUNNING"):
            return info
        time.sleep(poll_interval)


def trigger_index_and_wait(poll_interval: float = 60.0) -> dict:
    """Trigger generate-index and poll GET /arcana/ for the real result, ignoring the
    POST's own (expected-to-time-out) response."""
    _check_config()
    url = f"{config.ARCANA_BASE_URL}/arcana/{config.ARCANA_NAME}/generate-index"
    print("Triggering generate-index...")
    try:
        with httpx.Client(headers={"Authorization": config.ARCANA_TOKEN}, timeout=90) as client:
            client.post(url)
    except httpx.HTTPError as e:
        print(f"  (POST itself errored/timed out client-side -- expected: {e!r})")
    return wait_for_index(poll_interval=poll_interval)


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


def upload_and_index_in_batches(batch_size: int = config.ARCANA_INDEX_BATCH_SIZE, force: bool = False) -> None:
    """
    Upload + index in batches of `batch_size` papers, to stay under GWDG's
    observed ~6h generate-index ceiling.

    Stops immediately if a batch's index run doesn't come back INDEXED,
    rather than continuing on to further batches: a failed/timed-out run
    has been observed to wipe the *entire* existing index, not just fail
    to add the new batch, so continuing blindly after an error risks
    repeating that against whatever *did* just get indexed.
    """
    _check_config()
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS arcana_status VARCHAR DEFAULT 'pending'")

    ready = "(ingest_method = 'html' AND md_status = 'converted') OR (ingest_method = 'pdf' AND pdf_status = 'fetched')"
    where = ready if force else f"({ready}) AND arcana_status = 'pending'"
    todo = con.execute(f"SELECT bibcode, arxiv_id, ingest_method FROM papers WHERE {where}").fetchall()
    print(f"{len(todo)} papers to upload+index, in batches of {batch_size}")

    headers = {"Authorization": config.ARCANA_TOKEN, "accept": "application/json"}
    n_batches = (len(todo) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(todo), batch_size), start=1):
        batch = todo[i : i + batch_size]
        print(f"\n=== Batch {batch_num}/{n_batches}: papers {i + 1}-{i + len(batch)} of {len(todo)} ===")

        ok = fail = 0
        t_upload0 = time.time()
        with httpx.Client(headers=headers, timeout=300) as client:
            for bibcode, arxiv_id, ingest_method in batch:
                path, filename, content_type = _resolve_upload(arxiv_id, ingest_method)
                if not path.exists():
                    print(f"  [missing] {arxiv_id}")
                    con.execute("UPDATE papers SET arcana_status = 'failed' WHERE bibcode = ?", [bibcode])
                    fail += 1
                    continue
                try:
                    _post_file(client, path, filename, content_type)
                    con.execute("UPDATE papers SET arcana_status = 'uploaded' WHERE bibcode = ?", [bibcode])
                    ok += 1
                except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
                    print(f"  [fail] {arxiv_id}: {e}")
                    con.execute("UPDATE papers SET arcana_status = 'failed' WHERE bibcode = ?", [bibcode])
                    fail += 1
        print(f"Batch upload: {ok} ok, {fail} failed in {time.time() - t_upload0:.1f}s")

        t_index0 = time.time()
        info = trigger_index_and_wait()
        status = info.get("index_status")
        print(f"Batch index result: {status} in {time.time() - t_index0:.1f}s "
              f"(files_indexed={info.get('total_files_indexed')}, chunks_indexed={info.get('total_chunks_indexed')})")

        if status != "INDEXED":
            print(f"\n[ABORT] Batch {batch_num}/{n_batches} did not come back INDEXED "
                  f"(status={status}, error={info.get('error_msg')}). Stopping here rather than "
                  f"continuing to further batches -- check papers.duckdb.arcana_status for what's "
                  f"uploaded so far, and the live arcana state before retrying.")
            con.close()
            return

    con.close()
    print(f"\nAll {n_batches} batches uploaded and indexed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload paper(s) to Arcana and (re)index")
    parser.add_argument("--arxiv-id", help="Upload this one paper's converted Markdown")
    parser.add_argument("--all", action="store_true", help="Upload every converted, not-yet-uploaded paper")
    parser.add_argument("--batched", action="store_true",
                         help="Upload + index in batches of --batch-size, stopping on the first failed batch")
    parser.add_argument("--batch-size", type=int, default=config.ARCANA_INDEX_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="With --all/--batched, re-upload regardless of arcana_status")
    parser.add_argument("--no-index", action="store_true", help="With --all, skip triggering generate-index after upload")
    args = parser.parse_args()

    if args.batched:
        upload_and_index_in_batches(batch_size=args.batch_size, force=args.force)
    elif args.arxiv_id:
        con = duckdb.connect(str(config.DB_PATH))
        row = con.execute("SELECT ingest_method FROM papers WHERE arxiv_id = ?", [args.arxiv_id]).fetchone()
        con.close()
        upload_paper(args.arxiv_id, ingest_method=(row[0] if row else "html"))
        if not args.no_index:
            generate_index()
    elif args.all:
        upload_all(force=args.force)
        if not args.no_index:
            generate_index()

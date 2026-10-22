"""
Embed every chunk in `chunks` with config.EMBED_MODEL (a Qwen3-backbone embedder,
currently harrier-oss-v1-0.6b; run locally with MLX, bf16 -- 8-/4-bit saved memory
but were slower) and store the vectors in `chunk_embeddings` in papers.duckdb.

Embedded text for a chunk = paper title + heading path (+ blank line) + body,
where body is the chunk text, or for `kind='table'` just the caption + header
row (chunk.table_embed_text). `kind='boilerplate'` chunks are never embedded (they
stay reachable through `search grep`). Documents get no instruction prefix; queries (retrieval
stage) use config.QUERY_INSTRUCTION (embed.format_query). The vector is the final
hidden state of the appended <|endoftext|> token, L2-normalized, so cosine
similarity is a dot product.

`chunk_embeddings` is keyed (bibcode, chunk_index, model) and stores a hash of
the exact embedded text. That is the "done" test: a chunk is pending if it has
no row for this model or its stored hash differs from the current embedded
text's. So re-chunking (which reshuffles chunk_index), a changed prefix
(config.EMBED_TITLE_PREFIX) or an edited table rule all re-embed exactly the
affected chunks instead of leaving stale vectors attached to the wrong text. What
is pending is found by comparing those hashes, never by comparing counts: equal
counts say nothing about whether the stored vectors still match the text.
Rows whose (bibcode, chunk_index) no longer exists in `chunks`, and vectors of
chunks now tagged boilerplate, are deleted on start. Vectors are stored as FLOAT[] (any dimension, so several models can
coexist); cast with `embedding::FLOAT[1024]` for DuckDB's VSS/array functions.

Chunks are pulled in paper order, collected into windows of about
config.EMBED_COMMIT_TOKENS tokens, length-sorted inside each window (so
batches pad little), embedded, and committed in one transaction -- Ctrl-C
loses at most the window in flight, and re-running resumes.

Run directly (from the `embed` environment):
    pixi run embed
    pixi run embed --dry-run              # find what is pending, embed nothing
    pixi run embed --limit 200            # smoke-test on the first 200 pending chunks
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Iterator, NamedTuple

import duckdb
import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.models.qwen3 import Model, ModelArgs

# Only the tokenizer is used from transformers; without this it warns on every run
# that PyTorch is missing.
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

from transformers import AutoTokenizer, PreTrainedTokenizerBase

import config

from .chunk import table_embed_text

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    bibcode     VARCHAR,
    chunk_index INTEGER,
    model       VARCHAR,
    text_hash   VARCHAR,
    embedding   FLOAT[],
    PRIMARY KEY (bibcode, chunk_index, model)
);
"""

_EOT = '<|endoftext|>'
_INSERT_ROWS_PER_STATEMENT = 200


class Embedded(NamedTuple):
    vectors: np.ndarray  # (n, dim) float32, L2-normalized, in input order
    n_tokens: int
    n_truncated: int


_HF_PATTERNS = ['*.json', '*.safetensors', '*.txt']


def _weights_complete(path: Path) -> bool:
    index = path / 'model.safetensors.index.json'
    if index.exists():  # sharded checkpoint: every listed shard must be present
        shards = set(json.loads(index.read_text())['weight_map'].values())
        return all((path / s).exists() for s in shards)
    return any(path.glob('*.safetensors'))


def _model_dir(offline_first: bool) -> Path:
    repo = config.EMBED_MODEL
    if offline_first:
        # snapshot_download normally checks the Hub for a newer revision on
        # every call; a query CLI shouldn't need the network once cached.
        try:
            path = Path(
                snapshot_download(
                    repo, allow_patterns=_HF_PATTERNS, local_files_only=True
                )
            )
            if (path / 'config.json').exists() and _weights_complete(path):
                return path
        except Exception:  # not cached (or partial): fall through to the normal path
            pass
    return Path(snapshot_download(repo, allow_patterns=_HF_PATTERNS))


def _check_supported(path: Path) -> None:
    """Stop early on a model this loader would load wrongly: it builds a Qwen3 network
    and pools the last token."""
    model_type = json.loads((path / 'config.json').read_text()).get('model_type')
    if model_type != 'qwen3':
        raise SystemExit(
            f'{config.EMBED_MODEL} is a {model_type!r} model; only Qwen3-backbone '
            'embedding models are supported (see EMBED_MODEL in config.py)'
        )
    pooling = path / '1_Pooling' / 'config.json'
    if pooling.exists() and not json.loads(pooling.read_text()).get(
        'pooling_mode_lasttoken'
    ):
        raise SystemExit(
            f'{config.EMBED_MODEL} does not use last-token pooling, which is the only '
            'pooling supported (see EMBED_MODEL in config.py)'
        )


def load_embedder(offline_first: bool = False) -> tuple[Model, PreTrainedTokenizerBase]:
    path = _model_dir(offline_first)
    _check_supported(path)
    net = Model(ModelArgs.from_dict(json.loads((path / 'config.json').read_text())))
    weights: dict[str, mx.array] = {}
    for f in sorted(path.glob('*.safetensors')):
        weights.update(mx.load(str(f)))
    # The HF embedding checkpoints omit the "model." key prefix mlx_lm's
    # causal-LM wrapper expects (it is a plain Qwen3 backbone, no lm_head).
    net.load_weights(
        [
            (k if k.startswith(('model.', 'lm_head')) else f'model.{k}', v)
            for k, v in weights.items()
        ]
    )
    mx.eval(net.parameters())
    return net, AutoTokenizer.from_pretrained(str(path))


def embedded_text(
    title: str | None,
    heading: str | None,
    kind: str,
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """The exact string that gets embedded for a chunk. `heading` is the
    chunk's heading_path, falling back to its section_heading."""
    body = table_embed_text(text, tokenizer) if kind == 'table' else text
    prefix = '\n'.join(
        p for p in (title if config.EMBED_TITLE_PREFIX else None, heading) if p
    )
    return f'{prefix}\n\n{body}' if prefix else body


def format_query(query: str) -> str:
    """Counterpart of embedded_text() for retrieval: queries are wrapped in
    config.QUERY_INSTRUCTION, documents are embedded as they are."""
    return config.QUERY_INSTRUCTION.format(query=query)


def _batches(order: list[int], ids: list[list[int]]) -> Iterator[list[int]]:
    """Group length-sorted indices into batches capped by item count and by
    padded tokens (items x longest), so one long chunk can't blow up memory."""
    cur: list[int] = []
    for i in order:
        if cur and (
            len(cur) >= config.EMBED_MAX_BATCH
            or (len(cur) + 1) * len(ids[i]) > config.EMBED_BATCH_TOKENS
        ):
            yield cur
            cur = []
        cur.append(i)
    if cur:
        yield cur


def embed_texts(
    model: Model, tokenizer: PreTrainedTokenizerBase, texts: list[str]
) -> Embedded:
    eot = tokenizer.convert_tokens_to_ids(_EOT)
    cap = config.MAX_EMBED_TOKENS - 1
    raw = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    n_truncated = sum(len(r) > cap for r in raw)
    ids = [r[:cap] + [eot] for r in raw]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))

    out: list[np.ndarray | None] = [None] * len(ids)
    for batch in _batches(order, ids):
        longest = max(len(ids[i]) for i in batch)
        arr = mx.array([ids[i] + [eot] * (longest - len(ids[i])) for i in batch])
        hidden = model.model(
            arr
        )  # causal, so right-padding can't affect earlier positions
        last = mx.array([len(ids[i]) - 1 for i in batch])
        v = hidden[mx.arange(len(batch)), last].astype(mx.float32)
        v = v / mx.linalg.norm(v, axis=-1, keepdims=True)
        mx.eval(v)
        v = np.array(v)
        for j, i in enumerate(batch):
            out[i] = v[j]
    return Embedded(np.stack(out), sum(map(len, ids)), n_truncated)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def _insert_vectors(
    con: duckdb.DuckDBPyConnection,
    model_id: str,
    items: list[tuple[str, int, str, str]],
    vectors: np.ndarray,
) -> None:
    """Upsert (bibcode, chunk_index, hash) items with their vectors. Vectors go
    in as FLOAT[] SQL literals (%.9g round-trips float32 exactly) rather than
    bound parameters: converting 1024 Python floats per row through
    executemany cost ~40 ms/row, ~10x slower than the literal."""
    if not np.isfinite(vectors).all():
        raise ValueError('non-finite values in embeddings')
    for s in range(0, len(items), _INSERT_ROWS_PER_STATEMENT):
        part = range(s, min(s + _INSERT_ROWS_PER_STATEMENT, len(items)))
        sql = 'INSERT OR REPLACE INTO chunk_embeddings VALUES ' + ','.join(
            '(?, ?, ?, ?, [' + ','.join(f'{x:.9g}' for x in vectors[k]) + ']::FLOAT[])'
            for k in part
        )
        con.execute(
            sql,
            [
                p
                for k in part
                for p in (items[k][0], items[k][1], model_id, items[k][3])
            ],
        )


def _pending(
    con: duckdb.DuckDBPyConnection, model_id: str, tokenizer: PreTrainedTokenizerBase
) -> Iterator[tuple[str, int, str, str]]:
    """Yield (bibcode, chunk_index, embedded_text, hash) for every non-boilerplate
    chunk whose stored hash for this model is missing or stale, in paper order."""
    try:
        done = {
            (b, i): h
            for b, i, h in con.execute(
                'SELECT bibcode, chunk_index, text_hash FROM chunk_embeddings WHERE model = ?',
                [model_id],
            ).fetchall()
        }
    except duckdb.CatalogException:  # nothing embedded yet
        done = {}
    cur = con.cursor()  # separate connection, so window inserts don't disturb the scan
    cur.execute(
        """SELECT c.bibcode, c.chunk_index, c.kind, c.section_heading, c.heading_path, c.text, p.title
           FROM chunks c JOIN papers p USING (bibcode)
           WHERE c.kind <> 'boilerplate'
           ORDER BY c.bibcode, c.chunk_index"""
    )
    while rows := cur.fetchmany(2000):
        for bibcode, idx, kind, section_heading, heading_path, text, title in rows:
            et = embedded_text(
                title, heading_path or section_heading, kind, text, tokenizer
            )
            h = _hash(et)
            if done.get((bibcode, idx)) != h:
                yield bibcode, idx, et, h


# Tokens per second `embed` reached on the machine it was developed on; only used
# for the rough time estimate in --dry-run, real speed depends on your hardware.
_REFERENCE_TOKENS_PER_SECOND = 3500


def _dry_run(limit: int | None) -> None:
    """Report what a run would do. Read-only: no model is loaded, nothing is written."""
    if not config.DB_PATH.exists():
        raise SystemExit(f'No database at {config.DB_PATH}; run `pixi run query` first')
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    try:
        columns = {r[1] for r in con.execute("PRAGMA table_info('chunks')").fetchall()}
        if not {'kind', 'heading_path'} <= columns:
            raise SystemExit(
                'chunks table is missing or has the old schema; run `pixi run chunk --rebuild` first'
            )
        model_id = config.EMBED_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        n_pending = n_chars = 0
        for _, _, text, _ in _pending(con, model_id, tokenizer):
            n_pending += 1
            n_chars += len(text)
            if limit and n_pending >= limit:
                break
        try:
            orphans = con.execute(
                """SELECT count(*) FROM chunk_embeddings e WHERE NOT EXISTS
                   (SELECT 1 FROM chunks c WHERE c.bibcode = e.bibcode AND c.chunk_index = e.chunk_index)"""
            ).fetchone()[0]
            demoted = con.execute(
                """SELECT count(*) FROM chunk_embeddings e WHERE e.model = ? AND EXISTS
                   (SELECT 1 FROM chunks c WHERE c.bibcode = e.bibcode
                      AND c.chunk_index = e.chunk_index AND c.kind = 'boilerplate')""",
                [model_id],
            ).fetchone()[0]
        except duckdb.CatalogException:  # nothing embedded yet
            orphans = demoted = 0
    finally:
        con.close()

    n_tokens = int(n_chars / 3.5)  # same rough chars-per-token rule the run uses
    seconds = n_tokens / _REFERENCE_TOKENS_PER_SECOND
    waited = f'{seconds / 3600:.1f} h' if seconds >= 3600 else f'{seconds / 60:.0f} min'
    print(f'Dry run, nothing embedded. Model: {model_id}')
    print(
        f'  {n_pending:,} chunks pending'
        + (' (stopped counting at --limit)' if limit and n_pending >= limit else '')
    )
    print(f'  about {n_tokens / 1e6:.1f} million tokens')
    print(
        f'  roughly {waited} at {_REFERENCE_TOKENS_PER_SECOND:,} tokens/s, the speed on the development machine'
    )
    if orphans or demoted:
        print(
            f'  a real run would first delete {orphans:,} vectors whose chunk is gone '
            f'and {demoted:,} vectors of chunks now tagged boilerplate'
        )


def embed(limit: int | None = None, dry_run: bool = False) -> None:
    if dry_run:
        _dry_run(limit)
        return

    con = duckdb.connect(str(config.DB_PATH))
    try:
        columns = {r[1] for r in con.execute("PRAGMA table_info('chunks')").fetchall()}
        if not {'kind', 'heading_path'} <= columns:
            raise SystemExit(
                'chunks table is missing or has the old schema; run `pixi run chunk --rebuild` first'
            )
        con.execute(_SCHEMA)
        model_id = config.EMBED_MODEL

        orphans = con.execute(
            """DELETE FROM chunk_embeddings e WHERE NOT EXISTS
               (SELECT 1 FROM chunks c WHERE c.bibcode = e.bibcode AND c.chunk_index = e.chunk_index)
               RETURNING 1"""
        ).fetchall()
        if orphans:
            print(f'Deleted {len(orphans)} embeddings whose chunk no longer exists')

        # A chunk re-classified as boilerplate keeps its (bibcode, chunk_index), so it
        # isn't an orphan -- but it's no longer eligible, and search scores every row of
        # chunk_embeddings. Drop its vector for this model.
        demoted = con.execute(
            """DELETE FROM chunk_embeddings e WHERE e.model = ? AND EXISTS
               (SELECT 1 FROM chunks c WHERE c.bibcode = e.bibcode
                  AND c.chunk_index = e.chunk_index AND c.kind = 'boilerplate')
               RETURNING 1""",
            [model_id],
        ).fetchall()
        if demoted:
            print(f'Deleted {len(demoted)} embeddings of chunks now tagged boilerplate')

        eligible = con.execute(
            "SELECT count(*) FROM chunks WHERE kind <> 'boilerplate'"
        ).fetchone()[0]
        have = con.execute(
            'SELECT count(*) FROM chunk_embeddings WHERE model = ?', [model_id]
        ).fetchone()[0]
        # Only a rough size for the progress line: equal counts do not mean up to date
        # (a changed prefix or re-chunked text leaves every chunk with a stale vector).
        est_pending = max(eligible - have, 0)
        target = min(est_pending, limit) if limit else est_pending
        print(
            f'{model_id}: {have}/{eligible} chunks have a vector'
            + (f'; ~{est_pending} pending' if est_pending else '')
            + (f', stopping after {limit}' if limit else '')
        )

        # Scan with just the tokenizer first, so a run with nothing to do never loads the model.
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pending = _pending(con, model_id, tokenizer)
        first = next(pending, None)
        if first is None:
            print('Nothing pending: every vector matches its chunk text')
            return

        net, _ = load_embedder()
        pending = itertools.chain([first], pending)

        n_done = n_tokens = n_truncated = 0
        t0 = time.perf_counter()
        window: list[tuple[str, int, str, str]] = []
        window_est = 0

        def flush() -> None:
            nonlocal window, window_est, n_done, n_tokens, n_truncated
            if not window:
                return
            emb = embed_texts(net, tokenizer, [w[2] for w in window])
            con.begin()
            try:
                _insert_vectors(
                    con,
                    model_id,
                    [(b, i, et, h) for b, i, et, h in window],
                    emb.vectors,
                )
                con.commit()
            except BaseException:
                con.rollback()
                raise
            n_done += len(window)
            n_tokens += emb.n_tokens
            n_truncated += emb.n_truncated
            window, window_est = [], 0
            mx.clear_cache()
            dt = time.perf_counter() - t0
            progress = f'{n_done}/{target}' if target > n_done else f'{n_done}'
            eta = (
                f' | ETA {(target - n_done) / (n_done / dt) / 3600:.1f} h'
                if target > n_done
                else ''
            )
            print(f'  {progress} chunks | {n_tokens / dt:.0f} tok/s{eta}', flush=True)

        for item in pending:
            if limit and n_done + len(window) >= limit:
                break
            window.append(item)
            window_est += (
                len(item[2]) // 3
            )  # ~3.5 chars/token; only used to size the commit window
            if window_est >= config.EMBED_COMMIT_TOKENS:
                flush()
        flush()

        dt = time.perf_counter() - t0
        print(
            f'Embedded {n_done} chunks ({n_tokens} tokens, {n_truncated} truncated at '
            f'{config.MAX_EMBED_TOKENS}) in {dt / 60:.1f} min'
        )
    finally:
        con.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Embed chunks (MLX) into chunk_embeddings'
    )
    parser.add_argument(
        '--limit', type=int, default=None, help='Stop after embedding N pending chunks'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Find what is pending and estimate the time, embed nothing',
    )
    args = parser.parse_args()

    embed(limit=args.limit, dry_run=args.dry_run)

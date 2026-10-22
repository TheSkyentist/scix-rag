"""
Phase 2: re-embed the sub-corpus (evals.subset, data/eval_subset.duckdb) under chunking /
prefix / model variants and compare them on the Phase 1 item subset.

A variant is MODEL|CHUNKS|PREFIX:
  MODEL   an evals.embedders key (qwen0.6, qwen4, harrier0.6, jina5s)
  CHUNKS  base (the production chunks, copied) or a size N built from data/markdown (chunks_N)
  PREFIX  std      title + heading path (production)
          abstract title + the abstract's opening (~60 tokens) + heading path
          heading  heading path only (no title)
          prev512  title + the ~512 tokens of paper text preceding the chunk + heading path
                   (a causal stand-in for late chunking: the pooled last token has read
                   the preceding context)
Vectors go to data/eval_cache/phase2/<variant>/part_*.npz, written every ~250K tokens,
so an interrupted run resumes. qwen0.6|base|std is copied from papers.duckdb.

    .pixi/envs/embed/bin/python -m evals.phase2 chunk 512
    .pixi/envs/embed/bin/python -m evals.phase2 embed 'qwen0.6|base|abstract'
    .pixi/envs/embed/bin/python -m evals.phase2 eval 'qwen0.6|base|std' 'qwen0.6|base|abstract' ... [--out F]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

import config
from evals.run_retrieval_eval import SETS_DIR, Item, load_items, load_twins, report, score
from evals.subset import SUBSET_DB

OUT = config.DB_PATH.parent / "eval_cache" / "phase2"
SUBSET_FTS = config.DB_PATH.parent / "eval_subset_fts.duckdb"  # BM25 over chunks_base (separate file: no lock clash)
WINDOW_TOKENS = 250_000
DEPTH = 200


def _vdir(variant: str) -> Path:
    return OUT / variant.replace("|", "__")


# --- chunk variants -----------------------------------------------------------------------

def cmd_chunk(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from src import chunk as C

    config.MAX_CHUNK_TOKENS = args.max_tokens  # chunk_paper reads it at call time
    tok = AutoTokenizer.from_pretrained(C._TOKENIZER_NAME)
    con = duckdb.connect(str(SUBSET_DB))
    table = f"chunks_{args.max_tokens}"
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM chunks_base WHERE false")
    rows = []
    for bibcode, arxiv_id in con.execute("SELECT bibcode, arxiv_id FROM papers ORDER BY bibcode").fetchall():
        md = (config.MARKDOWN_DIR / f"{arxiv_id or bibcode}.md").read_text(encoding="utf-8")
        for i, c in enumerate(C.chunk_paper(md, tok)):
            rows.append((bibcode, arxiv_id, i, c.heading, c.heading_path, c.kind, c.text, c.n_tokens))
    con.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    n, t = con.execute(f"SELECT count(*), sum(n_tokens) FROM {table} WHERE kind <> 'boilerplate'").fetchone()
    print(f"{table}: {n} embeddable chunks, {t / 1e6:.1f}M tokens")


# --- embedding ----------------------------------------------------------------------------

def _texts(con: duckdb.DuckDBPyConnection, chunks: str, prefix: str, tok) -> list[tuple[str, int, str]]:
    from src.chunk import table_embed_text

    table = "chunks_base" if chunks == "base" else f"chunks_{chunks}"
    rows = con.execute(
        f"""SELECT c.bibcode, c.chunk_index, c.kind, coalesce(c.heading_path, c.section_heading), c.text,
                   p.title, coalesce(p.abstract, '')
            FROM {table} c JOIN papers p USING (bibcode) ORDER BY c.bibcode, c.chunk_index"""
    ).fetchall()
    out, prev_tail = [], {}
    by_paper: dict[str, list] = defaultdict(list)
    for r in rows:
        by_paper[r[0]].append(r)
    for b, rs in by_paper.items():
        running = ""  # paper text so far, for prev512
        for b_, i, kind, head, text, title, abstract in rs:
            body = table_embed_text(text, tok) if kind == "table" else text
            if kind != "boilerplate":
                if prefix == "std":
                    pre = [title, head]
                elif prefix == "heading":
                    pre = [head]
                elif prefix == "abstract":
                    ab = re.sub(r"\s+", " ", abstract).strip()
                    pre = [title, tok.decode(tok.encode(ab, add_special_tokens=False)[:60]) if ab else None, head]
                elif prefix == "prev512":
                    ids = tok.encode(running[-6000:], add_special_tokens=False)[-512:]
                    pre = [title, tok.decode(ids) if ids else None, head]
                else:
                    raise SystemExit(f"unknown prefix {prefix}")
                p = "\n".join(x for x in pre if x)
                out.append((b, i, f"{p}\n\n{body}" if p else body))
            running += "\n\n" + text
    return out


def cmd_embed(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from evals.embedders import Embedder

    model, chunks, prefix = args.variant.split("|")
    vdir = _vdir(args.variant)
    vdir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(SUBSET_DB), read_only=True)
    if args.variant == "qwen0.6|base|std":
        main = duckdb.connect(str(config.DB_PATH), read_only=True)
        bibs = [b for (b,) in con.execute("SELECT bibcode FROM papers").fetchall()]
        r = main.execute(
            """SELECT bibcode, chunk_index, embedding::FLOAT[1024] AS v FROM chunk_embeddings
               WHERE model = 'Qwen3-Embedding-0.6B' AND bibcode IN (SELECT unnest(?)) ORDER BY 1, 2""", [bibs]
        ).fetchnumpy()
        np.savez(vdir / "part_0000.npz", bibcode=np.array(r["bibcode"].tolist(), dtype=str), idx=r["chunk_index"].astype(np.int32), V=np.stack(r["v"]))
        print(f"copied {len(r['bibcode'])} production vectors")
        return
    emb = Embedder(model)
    todo = _texts(con, chunks, prefix, emb.tok)
    done = set()
    parts = sorted(vdir.glob("part_*.npz"))
    for f in parts:
        d = np.load(f)
        done |= set(zip(d["bibcode"].tolist(), d["idx"].tolist()))
    todo = [t for t in todo if (t[0], t[1]) not in done]
    print(f"{args.variant}: {len(done)} done, {len(todo)} to embed", flush=True)
    k, t0, ntok = len(parts), time.time(), 0
    i = 0
    while i < len(todo):
        win, wtok = [], 0
        while i < len(todo) and wtok < WINDOW_TOKENS:
            win.append(todo[i])
            wtok += len(todo[i][2]) // 3  # rough; exact count comes back from the model
            i += 1
        V, n = emb.docs([t[2] for t in win])
        mx.clear_cache()  # as src/embed.py: padded shapes vary, so MLX's buffer cache otherwise grows for hours
        ntok += n
        np.savez(vdir / f"part_{k:04d}.npz", bibcode=np.array([t[0] for t in win], dtype=str),
                 idx=np.array([t[1] for t in win], dtype=np.int32), V=V.astype(np.float32))
        k += 1
        el = time.time() - t0
        print(f"  {i}/{len(todo)} chunks, {ntok / el:.0f} tok/s, eta {(len(todo) - i) * el / i / 3600:.1f} h", flush=True)
    (vdir / "DONE").write_text(json.dumps({"tokens": ntok, "seconds": time.time() - t0}))


# --- BM25 over the sub-corpus -------------------------------------------------------------

def cmd_fts(args: argparse.Namespace) -> None:
    con = duckdb.connect(str(SUBSET_FTS))
    con.execute("INSTALL fts; LOAD fts")
    con.execute(f"ATTACH '{SUBSET_DB}' AS src (READ_ONLY)")
    con.execute("""CREATE OR REPLACE TABLE docs AS
        SELECT row_number() OVER (ORDER BY c.bibcode, c.chunk_index)::INTEGER AS id, c.bibcode, c.chunk_index, p.year,
               coalesce(c.heading_path, '') || ' ' || c.text AS body
        FROM src.chunks_base c JOIN src.papers p USING (bibcode) WHERE c.kind <> 'boilerplate'""")
    con.execute("DETACH src")
    con.execute("PRAGMA create_fts_index('docs', 'id', 'body', stemmer='english', stopwords='english', "
                "ignore='(\\.|[^a-z0-9])+', strip_accents=1, lower=1, overwrite=1)")
    print(con.execute("SELECT count(*) FROM docs").fetchone()[0], "docs indexed")


def _bm25(fts: duckdb.DuckDBPyConnection, query: str, since: int | None) -> list[tuple[str, int, float]]:
    sql = ("SELECT bibcode, chunk_index, score FROM (SELECT bibcode, chunk_index, year, "
           "fts_main_docs.match_bm25(id, ?) AS score FROM docs) WHERE score IS NOT NULL")
    params: list = [re.sub(r"[^\w\s\-]", " ", query)]
    if since is not None:
        sql += " AND year >= ?"
        params.append(since)
    return fts.execute(sql + " ORDER BY score DESC, bibcode, chunk_index LIMIT ?", params + [DEPTH]).fetchall()


# --- evaluation ---------------------------------------------------------------------------

class SubIndex:
    def __init__(self, variant: str, con: duckdb.DuckDBPyConnection):
        parts = sorted(_vdir(variant).glob("part_*.npz"))
        if not parts:
            raise SystemExit(f"no vectors for {variant}")
        ds = [np.load(f) for f in parts]
        self.bibcode = np.concatenate([d["bibcode"] for d in ds])
        self.chunk_index = np.concatenate([d["idx"] for d in ds]).astype(int)
        self.M = np.concatenate([d["V"] for d in ds])
        year = dict(con.execute("SELECT bibcode, year FROM papers").fetchall())
        self.year = np.array([year.get(b) or 0 for b in self.bibcode], dtype=float)

    def top(self, qvec: np.ndarray, n: int, since: int | None) -> list[tuple[str, int, float]]:
        s = self.M @ qvec
        if since is not None:
            s = np.where(self.year >= since, s, -np.inf)
        n = min(n, len(s) - 1)
        idx = np.argpartition(-s, n)[:n]
        idx = idx[np.lexsort((self.chunk_index[idx], self.bibcode[idx], -s[idx]))]
        return [(self.bibcode[i], int(self.chunk_index[i]), float(s[i])) for i in idx]


def _gold_chunk_map(con: duckdb.DuckDBPyConnection, chunks: str, items: list[Item]) -> dict[str, set]:
    """Set B gold is a production chunk; for another chunking, any variant chunk containing
    one of three ~80-char probes (start / middle / end) of the gold chunk's text counts."""
    if chunks == "base":
        return {it.id: it.gold_chunks for it in items}
    table = f"chunks_{chunks}"
    out = {}
    for it in items:
        gs = set()
        for b, i in it.gold_chunks:
            (text,) = con.execute("SELECT text FROM chunks_base WHERE bibcode = ? AND chunk_index = ?", [b, i]).fetchone()
            t = re.sub(r"\s+", " ", text)
            probes = [t[:80], t[len(t) // 2: len(t) // 2 + 80], t[-80:]]
            for j, vt in con.execute(f"SELECT chunk_index, text FROM {table} WHERE bibcode = ?", [b]).fetchall():
                vt = re.sub(r"\s+", " ", vt)
                if any(p and p in vt for p in probes):
                    gs.add((b, j))
        out[it.id] = gs
    return out


def cmd_eval(args: argparse.Namespace) -> None:
    from evals.embedders import Embedder
    from src import search as S

    con = duckdb.connect(str(SUBSET_DB), read_only=True)
    keep = set(json.loads((SETS_DIR / "phase1_subset.json").read_text())["ids"])
    items = [it for it in load_items(["A", "B", "C"]) if it.id in keep]
    load_twins(duckdb.connect(str(config.DB_PATH), read_only=True))
    results: dict[str, dict[str, dict]] = {}
    qcache: dict[str, dict[str, np.ndarray]] = {}
    fts = None
    for spec in args.variants:
        variant, _, w = spec.partition("+bm25@")
        bm25_w = float(w) if w else None
        model, chunks, prefix = variant.split("|")
        if bm25_w is not None:
            if chunks != "base":
                raise SystemExit("BM25 index covers chunks_base only")
            if fts is None:
                fts = duckdb.connect(str(SUBSET_FTS), read_only=True)
                fts.execute("LOAD fts")
        if model not in qcache:
            emb = Embedder(model)
            qs = sorted({q for it in items for q in it.queries})
            qcache[model] = dict(zip(qs, emb.queries(qs)))
            del emb
        idx = SubIndex(variant, con)
        gmap = _gold_chunk_map(con, chunks, items)
        results[spec] = {}
        for it in items:
            t = time.perf_counter()
            lists = [idx.top(qcache[model][q], DEPTH, it.since) for q in it.queries]
            bm = [_bm25(fts, q, it.since) for q in it.queries] if bm25_w is not None else None
            ranked = [(h.bibcode, h.idx) for h in S._fuse(lists, DEPTH, DEPTH, bm, bm25_w)]
            it2 = Item(**{**it.__dict__, "gold_chunks": gmap[it.id]})
            results[spec][it.id] = {"m": score(it2, ranked), "t": time.perf_counter() - t}
        print(f"evaluated {spec} ({len(idx.bibcode)} vectors)", flush=True)
    text = "# Phase 2 eval (sub-corpus, Phase 1 item subset)\n" + report(results, items, args.variants[0], args.by)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunk")
    c.add_argument("max_tokens", type=int)
    e = sub.add_parser("embed")
    e.add_argument("variant")
    v = sub.add_parser("eval")
    v.add_argument("variants", nargs="+", help="first one is the baseline; append +bm25@W to fuse BM25 at weight W")
    v.add_argument("--by", default=None)
    v.add_argument("--out", default=None)
    sub.add_parser("fts")
    args = ap.parse_args()
    {"chunk": cmd_chunk, "embed": cmd_embed, "eval": cmd_eval, "fts": cmd_fts}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

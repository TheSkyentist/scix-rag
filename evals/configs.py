"""
Phase 1 retrieval configurations for evals.run_retrieval_eval (no re-index needed).

Each config maps (Ctx, Item) -> ranked [(bibcode, chunk_index)]. Extra resources are
loaded lazily and cached on the Ctx:
  BM25    data/eval_fts.duckdb (a copy of the non-boilerplate chunk text with a DuckDB
          `fts` index; built once, papers.duckdb is only attached read-only for it)
  HyDE    evals/sets/hyde_passages.jsonl (Claude-written hypothetical answers, Phase 1
          subset only; items without one fall back to the plain query lists)
  rerank  Qwen3-Reranker via evals.rerank
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np

import config
from evals.run_retrieval_eval import DEPTH, SETS_DIR, Ctx, Item, Ranked, _rrf, cfg_dense

FTS_DB = config.DB_PATH.parent / "eval_fts.duckdb"
RERANK_DEPTH = 100

INSTRUCTIONS = {
    "inst:generic": "Given a web search query, retrieve relevant passages that answer the query",
    "inst:evidence": (
        "Given a research question about astrophysics, retrieve passages from astronomy papers that contain "
        "the measurements, observations, models or arguments that answer it"
    ),
}


def _dense_lists(ctx: Ctx, it: Item, instruction: str = config.QUERY_INSTRUCTION):
    return [ctx.index.top(v, DEPTH, it.since) for v in ctx.qvecs(it.queries, instruction)]


# --- BM25 -----------------------------------------------------------------------------

def _fts(ctx: Ctx) -> duckdb.DuckDBPyConnection:
    if not hasattr(ctx, "_fts"):
        if not FTS_DB.exists():
            raise SystemExit(f"{FTS_DB} missing: build the BM25 index first (see evals/README or Phase 1 notes)")
        ctx._fts = duckdb.connect(str(FTS_DB), read_only=True)
        ctx._fts.execute("LOAD fts")
    return ctx._fts


def bm25_top(ctx: Ctx, query: str, n: int, since: int | None) -> list[tuple[str, int, float]]:
    q = re.sub(r"[^\w\s\-]", " ", query)
    sql = ("SELECT bibcode, chunk_index, score FROM (SELECT bibcode, chunk_index, year, "
           "fts_main_docs.match_bm25(id, ?) AS score FROM docs) WHERE score IS NOT NULL")
    params: list = [q]
    if since is not None:
        sql += " AND year >= ?"
        params.append(since)
    sql += " ORDER BY score DESC, bibcode, chunk_index LIMIT ?"
    params.append(n)
    return _fts(ctx).execute(sql, params).fetchall()


def cfg_bm25(ctx: Ctx, it: Item) -> Ranked:
    return _rrf([bm25_top(ctx, q, DEPTH, it.since) for q in it.queries])


def cfg_dense_bm25(ctx: Ctx, it: Item, weight: float = 1.0) -> Ranked:
    """Dense lists plus BM25 lists in one RRF; BM25 lists carry `weight` (passed through
    search._fuse's term-list slot, which only scales their RRF contribution)."""
    bm = [bm25_top(ctx, q, DEPTH, it.since) for q in it.queries]
    if weight == 1.0:
        return _rrf(_dense_lists(ctx, it) + bm)
    from src import search as S

    return [(h.bibcode, h.idx) for h in S._fuse(_dense_lists(ctx, it), DEPTH, DEPTH, bm, weight)]


# --- HyDE --------------------------------------------------------------------------------

def _hyde(ctx: Ctx) -> dict[str, str]:
    if not hasattr(ctx, "_hyde"):
        f = SETS_DIR / "hyde_passages.jsonl"
        ctx._hyde = {d["id"]: d["passage"] for d in map(json.loads, f.read_text().splitlines())} if f.exists() else {}
    return ctx._hyde


def _hyde_vec(ctx: Ctx, passage: str) -> np.ndarray:
    key = ("<doc>", passage)
    if key not in ctx._cache:
        ctx._cache[key] = ctx._embed_texts(ctx._model, ctx._tok, [passage]).vectors[0]  # embedded as a document
    return ctx._cache[key]


def cfg_dense_hyde(ctx: Ctx, it: Item) -> Ranked:
    lists = _dense_lists(ctx, it)
    if it.id in _hyde(ctx):
        lists.append(ctx.index.top(_hyde_vec(ctx, _hyde(ctx)[it.id]), DEPTH, it.since))
    return _rrf(lists)


# --- reranking ---------------------------------------------------------------------------

def _reranker(ctx: Ctx, size: str):
    key = f"_rr_{size}"
    if not hasattr(ctx, key):
        from evals.rerank import Reranker

        setattr(ctx, key, Reranker(size))
    return getattr(ctx, key)


RR_SCORES = config.DB_PATH.parent / "eval_cache" / "rerank_scores_{size}.jsonl"


def _rr_scores(ctx: Ctx, it: Item, head: Ranked, size: str) -> np.ndarray:
    """Reranker scores, shape (n query variants, len(head)). Every (query, chunk) score is
    cached on disk, so blending variants can be compared without re-scoring."""
    key = f"_rrcache_{size}"
    path = Path(str(RR_SCORES).format(size=size))
    if not hasattr(ctx, key):
        cache = {}
        if path.exists():
            for line in path.read_text().splitlines():
                d = json.loads(line)
                cache[(d["q"], d["b"], d["i"])] = d["s"]
        setattr(ctx, key, cache)
    cache = getattr(ctx, key)
    todo = [x for x in head if any((q, *x) not in cache for q in it.queries)]
    if todo:
        rows = ctx.con.execute(
            """SELECT c.bibcode, c.chunk_index, p.title, coalesce(c.heading_path, c.section_heading, ''), c.text
               FROM chunks c JOIN papers p USING (bibcode)
               WHERE (c.bibcode, c.chunk_index) IN (SELECT unnest(?), unnest(?))""",
            [[b for b, _ in todo], [i for _, i in todo]],
        ).fetchall()
        text = {(b, i): f"{t or ''}\n{h}\n\n{x}" for b, i, t, h, x in rows}
        with path.open("a") as f:
            for q in it.queries:
                need = [x for x in todo if (q, *x) not in cache]
                for x, sc in zip(need, _reranker(ctx, size).score(q, [text.get(x, "") for x in need])):
                    cache[(q, *x)] = sc
                    f.write(json.dumps({"q": q, "b": x[0], "i": x[1], "s": sc}) + "\n")
    return np.array([[cache[(q, *x)] for x in head] for q in it.queries])


def _rerank(
    ctx: Ctx, it: Item, ranked: Ranked, size: str, depth: int = RERANK_DEPTH,
    agg: str = "first", mix: tuple[str, float] | None = None,
) -> Ranked:
    """Re-sort the top `depth` chunks by reranker score. agg: which query variants score a
    chunk (first = the user's wording only, max/mean over all variants). mix: None replaces
    the fused order; ("rrf", w) fuses the original rank with the reranker rank (reranker
    weight w); ("lin", lam) blends lam * score + (1 - lam) * rank-normalised original."""
    head, tail = ranked[:depth], ranked[depth:]
    if not head:
        return ranked
    S = _rr_scores(ctx, it, head, size)
    s = {"first": S[0], "max": S.max(axis=0), "mean": S.mean(axis=0)}[agg]
    n = len(head)
    if mix is None:
        key = -s
    elif mix[0] == "rrf":
        rr_rank = np.empty(n)
        rr_rank[np.argsort(-s, kind="stable")] = np.arange(1, n + 1)
        key = -(1 / (60 + np.arange(1, n + 1)) + mix[1] / (60 + rr_rank))
    else:
        key = -(mix[1] * s + (1 - mix[1]) * (1 - np.arange(n) / max(n - 1, 1)))
    order = np.argsort(key, kind="stable")
    return [head[k] for k in order] + tail


def _rr06(agg: str, mix=None):
    return lambda ctx, it: _rerank(ctx, it, cfg_dense_bm25(ctx, it, 0.3), "0.6B", agg=agg, mix=mix)


CONFIGS = {
    **{name: (lambda instr: (lambda ctx, it: _rrf(_dense_lists(ctx, it, instr))))(instr) for name, instr in INSTRUCTIONS.items()},
    "bm25": cfg_bm25,
    "dense+bm25": cfg_dense_bm25,
    "dense+bm25@0.5": lambda ctx, it: cfg_dense_bm25(ctx, it, 0.5),
    "dense+bm25@0.3": lambda ctx, it: cfg_dense_bm25(ctx, it, 0.3),
    "dense+hyde": cfg_dense_hyde,
    "dense+term+bm25": lambda ctx, it: _merge_term(ctx, it),
    "rr0.6:dense": lambda ctx, it: _rerank(ctx, it, cfg_dense(ctx, it), "0.6B"),
    "rr0.6:dense+bm25": lambda ctx, it: _rerank(ctx, it, cfg_dense_bm25(ctx, it), "0.6B"),
    "rr0.6:dense+bm25@0.3": lambda ctx, it: _rerank(ctx, it, cfg_dense_bm25(ctx, it, 0.3), "0.6B"),
    # reranker blending variants (candidates: dense + BM25@0.3, top 100)
    "rr:max": _rr06("max"),
    "rr:mean": _rr06("mean"),
    "rr:first+rrf1": _rr06("first", ("rrf", 1.0)),
    "rr:max+rrf1": _rr06("max", ("rrf", 1.0)),
    "rr:max+rrf0.5": _rr06("max", ("rrf", 0.5)),
    "rr:max+lin0.5": _rr06("max", ("lin", 0.5)),
    "rr:max+lin0.3": _rr06("max", ("lin", 0.3)),
    "dense+term+bm25@0.3": lambda ctx, it: _merge_term(ctx, it, 0.3),
    "rr:max+rrf1:term+bm25@0.3": lambda ctx, it: _rerank(ctx, it, _merge_term(ctx, it, 0.3), "0.6B", agg="max", mix=("rrf", 1.0)),
    "rr4:dense": lambda ctx, it: _rerank(ctx, it, cfg_dense(ctx, it), "4B"),
    "rr4:dense+bm25@0.3": lambda ctx, it: _rerank(ctx, it, cfg_dense_bm25(ctx, it, 0.3), "4B"),
}


def _merge_term(ctx: Ctx, it: Item, bm25_weight: float = 1.0) -> Ranked:
    """dense+term fusion with the BM25 lists added at `bm25_weight` (falls back to
    dense+bm25 for items without --term regexes)."""
    from src import search as S

    if not it.terms:
        return cfg_dense_bm25(ctx, it, bm25_weight)
    qv = [v.tolist() for v in ctx.qvecs(it.queries)]
    pattern, _ = S._compile_terms(it.terms, False)
    scan = S._term_scan(ctx.con, ctx.model_id, qv, pattern, False, it.since, DEPTH)
    bm = [bm25_top(ctx, q, DEPTH, it.since) for q in it.queries]
    # one weighted RRF over: dense lists (1), term lists (TERM_WEIGHT), BM25 lists (bm25_weight)
    rrf: dict[tuple[str, int], float] = {}
    for lists, w in ((_dense_lists(ctx, it), 1.0), (scan.ranked, S.TERM_WEIGHT), (bm, bm25_weight)):
        for rows in lists:
            for rank, (b, i, _) in enumerate(rows, 1):
                rrf[(b, i)] = rrf.get((b, i), 0.0) + w / (S.RRF_K + rank)
    return sorted(rrf, key=lambda k: (-rrf[k], k))[:DEPTH]

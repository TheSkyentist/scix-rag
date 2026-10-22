"""
Retrieval eval over Sets A/B/C with paired statistics between configurations.

Sets (evals/sets/): A = citation-context questions (gold: the cited paper; the citing
paper is excluded from the ranking), B = passage questions (gold: one chunk and its
paper), C = the hand-curated cases in evals/search_gold.json plus the user's questions
in setC.jsonl (multi-variant, --since, --term; graded gold 2 = substantial, 1 = partial,
from pooling + Claude judging + user confirmation). Every configuration maps an item to a deep ranked list of chunks; papers are
ranked by first appearance. Metrics: paper Recall@{5,10,20,50} (found / min(k, #gold),
so many-gold questions can reach 1), paper nDCG@10 (graded gain 2^rel - 1), chunk
MRR@100 (Set B only). Each config is compared with the baseline, paired by item:
95% bootstrap CI on the mean difference, two-sided sign-flip permutation p, Holm-
adjusted across the compared configs (per set and metric).

Dense scoring runs in numpy on all vectors loaded once (same inner product as
search._top_chunks; `--parity N` checks the two agree). Read-only on the DB:

    .pixi/envs/embed/bin/python -m evals.run_retrieval_eval [--sets A,B,C] [--configs dense,...]
        [--baseline dense] [--by bucket] [--limit N] [--out results.md]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np

import config
from src import search as S

SETS_DIR = Path(__file__).parent / "sets"
GOLD_C = Path(__file__).parent / "search_gold.json"
DEPTH = 200  # chunks per ranked list
KS = (5, 10, 20, 50)
METRICS = [f"R@{k}" for k in KS] + ["nDCG@10", "MRR"]
# Twin rows (same paper as preprint + journal record not yet merged by ADS/canonicalize)
# count as one paper when scoring: bibcode -> representative, filled by load_twins().
TWINS: dict[str, str] = {}


def load_twins(con: duckdb.DuckDBPyConnection) -> int:
    rows = con.execute(
        """SELECT list(bibcode ORDER BY bibcode) FROM papers WHERE title IS NOT NULL
           GROUP BY lower(regexp_replace(title, '[^A-Za-z0-9]', '', 'g')), lower(split_part(first_author, ',', 1))
           HAVING count(*) > 1"""
    ).fetchall()
    for (group,) in rows:
        for b in group:
            TWINS[b] = group[0]
    return len(rows)


def _canon(b: str) -> str:
    return TWINS.get(b, b)


@dataclass
class Item:
    id: str
    set: str
    queries: list[str]
    gold_papers: set[str]
    gold_chunks: set[tuple[str, int]] = field(default_factory=set)
    exclude: set[str] = field(default_factory=set)
    since: int | None = None
    terms: list[str] = field(default_factory=list)
    tags: dict = field(default_factory=dict)
    rel: dict[str, int] = field(default_factory=dict)  # graded gold; empty = every gold paper counts 1


def load_items(sets: list[str]) -> list[Item]:
    items = []
    for s in sets:
        if s == "C":
            for c in json.loads(GOLD_C.read_text())["cases"]:
                items.append(Item(c["id"], "C", c["queries"], set(c["gold"]), since=c.get("since"),
                                  terms=c.get("terms") or [], tags={"status": c.get("status")}))
            if (SETS_DIR / "setC.jsonl").exists():
                for line in (SETS_DIR / "setC.jsonl").read_text().splitlines():
                    d = json.loads(line)
                    rel = {b: r for b, r in d["rel"].items() if r > 0}
                    items.append(Item(d["id"], "C", d["queries"], set(rel), since=d.get("since"),
                                      terms=d.get("terms") or [], tags={"status": "user"}, rel=rel))
            continue
        for line in (SETS_DIR / f"set{s}.jsonl").read_text().splitlines():
            d = json.loads(line)
            items.append(Item(d["id"], d["set"], d["queries"], set(d["gold_papers"]),
                              {tuple(x) for x in d.get("gold_chunks", [])}, set(d.get("exclude", [])),
                              tags=d.get("tags", {})))
    return items


class DenseIndex:
    """All of one model's chunk vectors in memory, row-aligned with (bibcode, chunk_index)."""

    def __init__(self, con: duckdb.DuckDBPyConnection, model_id: str):
        (dim,) = con.execute("SELECT len(embedding) FROM chunk_embeddings WHERE model = ? LIMIT 1", [model_id]).fetchone()
        r = con.execute(
            f"""SELECT e.bibcode, e.chunk_index, p.year, e.embedding::FLOAT[{dim}] AS v
                FROM chunk_embeddings e JOIN papers p USING (bibcode)
                WHERE e.model = ? ORDER BY e.bibcode, e.chunk_index""",
            [model_id],
        ).fetchnumpy()
        self.bibcode = r["bibcode"]
        self.chunk_index = r["chunk_index"].astype(int)
        self.year = np.asarray(r["year"], dtype=float)
        self.M = np.stack(r["v"])

    def top(self, qvec: np.ndarray, n: int, since: int | None = None) -> list[tuple[str, int, float]]:
        s = self.M @ qvec
        if since is not None:
            s = np.where(self.year >= since, s, -np.inf)
        idx = np.argpartition(-s, n)[:n]
        idx = idx[np.lexsort((self.chunk_index[idx], self.bibcode[idx], -s[idx]))]
        return [(self.bibcode[i], int(self.chunk_index[i]), float(s[i])) for i in idx]


class Ctx:
    """Shared state for configs: DB, dense index, and a cached query embedder."""

    def __init__(self, con: duckdb.DuckDBPyConnection, model: str):
        from src.embed import embed_texts, load_embedder

        self.con, self.size = con, model
        self.model_id = model
        self.index = DenseIndex(con, self.model_id)
        self._sep = config.EMBED_MODELS[model][1]
        self._model, self._tok = load_embedder(model, offline_first=True)
        self._embed_texts = embed_texts
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def qvecs(self, queries: list[str], instruction: str = config.QUERY_INSTRUCTION) -> list[np.ndarray]:
        todo = [q for q in queries if (instruction, q) not in self._cache]
        if todo:
            v = self._embed_texts(self._model, self._tok, [f"Instruct: {instruction}\nQuery:{self._sep}{q}" for q in todo]).vectors
            self._cache.update({(instruction, q): v[i] for i, q in enumerate(todo)})
        return [self._cache[(instruction, q)] for q in queries]

    def prefetch(self, items: list[Item], instruction: str = config.QUERY_INSTRUCTION) -> None:
        self.qvecs(sorted({q for it in items for q in it.queries}), instruction)


Ranked = list[tuple[str, int]]


def _rrf(lists: list[list[tuple[str, int, float]]], term_lists=None) -> Ranked:
    hits = S._fuse(lists, DEPTH, DEPTH, term_lists)
    return [(h.bibcode, h.idx) for h in hits]


def cfg_dense(ctx: Ctx, it: Item) -> Ranked:
    """Production semantic search: every query variant, RRF-fused (no --term)."""
    return _rrf([ctx.index.top(v, DEPTH, it.since) for v in ctx.qvecs(it.queries)])


def cfg_dense_term(ctx: Ctx, it: Item) -> Ranked:
    """Production fused search: + the term-restricted rankings when the item has --term regexes."""
    lists = [ctx.index.top(v, DEPTH, it.since) for v in ctx.qvecs(it.queries)]
    if not it.terms:
        return _rrf(lists)
    pattern, _ = S._compile_terms(it.terms, False)
    scan = S._term_scan(ctx.con, ctx.model_id, [v.tolist() for v in ctx.qvecs(it.queries)], pattern, False, it.since, DEPTH)
    return _rrf(lists, scan.ranked)


CONFIGS: dict[str, Callable[[Ctx, Item], Ranked]] = {
    "dense": cfg_dense,
    "dense+term": cfg_dense_term,
}


def score(it: Item, ranked: Ranked) -> dict[str, float | None]:
    exclude = {_canon(b) for b in it.exclude}
    gold = {_canon(b) for b in it.gold_papers}
    ranked = [x for x in ranked if _canon(x[0]) not in exclude]
    papers = list(dict.fromkeys(_canon(b) for b, _ in ranked))
    out: dict[str, float | None] = {
        f"R@{k}": len(gold & set(papers[:k])) / min(k, len(gold)) for k in KS
    }
    gain: dict[str, int] = {}
    for b in it.gold_papers:
        gain[_canon(b)] = max(gain.get(_canon(b), 0), 2 ** it.rel.get(b, 1) - 1)
    dcg = sum(gain.get(b, 0) / np.log2(r + 2) for r, b in enumerate(papers[:10]))
    idcg = sum(g / np.log2(r + 2) for r, g in enumerate(sorted(gain.values(), reverse=True)[:10]))
    out["nDCG@10"] = dcg / idcg
    out["MRR"] = None
    if it.gold_chunks:
        out["MRR"] = next((1 / r for r, x in enumerate(ranked[:100], 1) if x in it.gold_chunks), 0.0)
    return out


def _paired(d: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float, float]:
    """95% bootstrap CI of mean(d) and two-sided sign-flip permutation p."""
    boots = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    p = (1 + np.sum(null >= abs(d.mean()) - 1e-12)) / (n + 1)
    return float(lo), float(hi), float(p)


def _holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def report(results: dict[str, dict[str, dict]], items: list[Item], baseline: str, by: str | None) -> str:
    rng = np.random.default_rng(0)
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        groups[it.set].append(it)
        if by and by in it.tags:
            groups[f"{it.set}:{by}={it.tags[by]}"].append(it)
    if len({it.set for it in items}) > 1:
        ab = [it for it in items if it.set in ("A", "B")]
        if ab:
            groups["A+B"] = ab
    lines = []
    configs = list(results)
    for g in sorted(groups, key=lambda s: (len(s), s)):
        its = groups[g]
        lines.append(f"\n### {g}  (n={len(its)})\n")
        mets = [m for m in METRICS if all(results[baseline][it.id]["m"][m] is not None for it in its)]
        lines.append("| config | " + " | ".join(mets) + " | s/query |")
        lines.append("|---" * (len(mets) + 2) + "|")
        cmp: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        for m in mets:
            others = [c for c in configs if c != baseline]
            stats = []
            for c in others:
                d = np.array([results[c][it.id]["m"][m] - results[baseline][it.id]["m"][m] for it in its])
                stats.append((d.mean(), *_paired(d, rng)))
            for c, (mean, lo, hi, p), padj in zip(others, stats, _holm([s[3] for s in stats]) if stats else []):
                cmp[(c, m)] = (mean, lo, hi, padj)
        for c in configs:
            cells = []
            for m in mets:
                v = np.mean([results[c][it.id]["m"][m] for it in its])
                if c == baseline or (c, m) not in cmp:
                    cells.append(f"{v:.3f}")
                else:
                    mean, lo, hi, p = cmp[(c, m)]
                    star = "*" if p < 0.05 else ""
                    cells.append(f"{v:.3f} ({mean:+.3f} [{lo:+.3f},{hi:+.3f}]{star})")
            t = np.mean([results[c][it.id]["t"] for it in its])
            lines.append(f"| {c}{' (base)' if c == baseline else ''} | " + " | ".join(cells) + f" | {t:.2f} |")
    lines.append("\nCells: mean (difference vs baseline [95% bootstrap CI], * = Holm-adjusted permutation p < 0.05).")
    return "\n".join(lines)


def check_parity(ctx: Ctx, items: list[Item], n: int) -> None:
    bad = 0
    for it in items[:n]:
        v = ctx.qvecs(it.queries[:1])[0]
        a = [(b, i) for b, i, _ in ctx.index.top(v, 50)]
        b = [(b, i) for b, i, _ in S._top_chunks(ctx.con, ctx.model_id, v.tolist(), 50, None)]
        if a != b:
            bad += 1
            print(f"parity mismatch on {it.id}: overlap {len(set(a) & set(b))}/50")
    print(f"parity: {n - bad}/{n} identical top-50 lists (numpy vs DuckDB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--configs", default="dense")
    ap.add_argument("--baseline", default="dense")
    ap.add_argument("--by", default=None, help="also split results by this tag (e.g. bucket, specificity, qtype)")
    ap.add_argument("--limit", type=int, default=None, help="first N items per set (smoke test)")
    ap.add_argument("--subset", action="store_true", help="only the Phase 1 subset (sets/phase1_subset.json)")
    ap.add_argument("--fresh", action="store_true", help="recompute configs even if cached in results/cache/")
    ap.add_argument("--model", default=config.EMBED_MODEL, choices=list(config.EMBED_MODELS))
    ap.add_argument("--parity", type=int, default=0, help="check numpy vs DuckDB top-50 on N items and exit")
    ap.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = ap.parse_args()

    from evals.configs import CONFIGS as EXTRA

    CONFIGS.update(EXTRA)
    names = args.configs.split(",")
    if args.baseline not in names:
        names.insert(0, args.baseline)
    unknown = [c for c in names if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; have {sorted(CONFIGS)}")
    items = load_items(args.sets.split(","))
    if args.limit:
        per: dict[str, int] = defaultdict(int)
        items = [it for it in items if (per.__setitem__(it.set, per[it.set] + 1) or per[it.set] <= args.limit)]
    if args.subset:
        keep = set(json.loads((SETS_DIR / "phase1_subset.json").read_text())["ids"])
        items = [it for it in items if it.id in keep]
    con = S._connect(args.db)
    print(f"{load_twins(con)} twin groups (same title + first author) scored as one paper")
    t0 = time.perf_counter()
    ctx = Ctx(con, args.model)
    print(f"loaded {len(ctx.index.bibcode)} vectors + model in {time.perf_counter() - t0:.1f}s; {len(items)} items")
    ctx.prefetch(items)
    if args.parity:
        check_parity(ctx, items, args.parity)
        return 0

    # Rankings are cached per (model, config, item) so slow configs (rerankers) survive a
    # re-run; scores are recomputed from the cached ranking, so gold edits still apply.
    cache_dir = Path(__file__).parent / "results" / "cache" / ctx.model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict]] = {}
    for c in names:
        cf = cache_dir / f"{c.replace('/', '_').replace(':', '_')}.json"
        cache = {} if args.fresh or not cf.exists() else json.loads(cf.read_text())
        results[c] = {}
        n_new = 0
        for k, it in enumerate(items):
            key = f"{it.id}|{'|'.join(it.queries)}"
            if key not in cache:
                t = time.perf_counter()
                cache[key] = {"ranked": CONFIGS[c](ctx, it)[:DEPTH], "t": time.perf_counter() - t}
                n_new += 1
                if n_new % 25 == 0:
                    cf.write_text(json.dumps(cache))
                    print(f"  {c}: {k + 1}/{len(items)}", flush=True)
            r = cache[key]
            results[c][it.id] = {"m": score(it, [tuple(x) for x in r["ranked"]]), "t": r["t"]}
        if n_new:
            cf.write_text(json.dumps(cache))
        print(f"ran {c} ({n_new} computed, {len(items) - n_new} cached)", flush=True)
    text = f"# Retrieval eval: sets {args.sets}, model {ctx.model_id}\n" + report(results, items, args.baseline, args.by)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

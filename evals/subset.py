"""
Phase 2 sub-corpus: a ~1,300-paper copy of the corpus on which chunking / prefix / model
variants are re-embedded and compared (data/eval_subset.duckdb; papers.duckdb is only
attached read-only).

Papers = every gold paper of Sets A/B/C, plus hard negatives (the top non-gold papers each
item retrieves with the current dense search and with BM25, from the Phase 1 ranking
cache), plus a random sample for breadth. Hard negatives keep the small corpus from being
trivially easy.

    .pixi/envs/embed/bin/python -m evals.subset select [--neg-dense 5 --neg-bm25 3 --random 150]
    .pixi/envs/embed/bin/python -m evals.subset chunk --max-tokens 512      # chunks_512 from data/markdown
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import duckdb

import config
from evals.run_retrieval_eval import load_items

SUBSET_DB = config.DB_PATH.parent / "eval_subset.duckdb"
CACHE = Path(__file__).parent / "results" / "cache" / "Qwen3-Embedding-0.6B"


def _top_papers(cache_file: Path, item_id: str, n: int, skip: set[str]) -> list[str]:
    d = json.loads(cache_file.read_text())
    for k, v in d.items():
        if k.split("|")[0] == item_id:
            out = []
            for b, _ in v["ranked"]:
                if b not in skip and b not in out:
                    out.append(b)
                if len(out) >= n:
                    break
            return out
    return []


def select(args: argparse.Namespace) -> None:
    items = load_items(["A", "B", "C"])
    if args.subset_items:
        keep = set(json.loads((Path(__file__).parent / "sets" / "phase1_subset.json").read_text())["ids"])
        items = [it for it in items if it.id in keep]
    gold = set().union(*[it.gold_papers for it in items])
    dense = {k.split("|")[0]: v["ranked"] for k, v in json.loads((CACHE / "dense.json").read_text()).items()}
    bm25 = {k.split("|")[0]: v["ranked"] for k, v in json.loads((CACHE / "bm25.json").read_text()).items()}

    def top(ranked, n, skip):
        out = []
        for b, _ in ranked:
            if b not in skip and b not in out:
                out.append(b)
            if len(out) >= n:
                break
        return out

    neg = set()
    for it in items:
        skip = it.gold_papers | it.exclude
        neg |= set(top(dense.get(it.id, []), args.neg_dense, skip)) | set(top(bm25.get(it.id, []), args.neg_bm25, skip))
    neg -= gold
    con = duckdb.connect(str(SUBSET_DB))
    con.execute(f"ATTACH '{config.DB_PATH}' AS src (READ_ONLY)")
    chunked = {b for (b,) in con.execute("SELECT DISTINCT bibcode FROM src.chunks").fetchall()}
    rest = sorted(chunked - gold - neg)
    rnd = set(random.Random(args.seed).sample(rest, args.random))
    papers = sorted((gold | neg | rnd) & chunked)
    con.execute("CREATE OR REPLACE TABLE papers AS SELECT * FROM src.papers WHERE bibcode IN (SELECT unnest(?))", [papers])
    con.execute("CREATE OR REPLACE TABLE chunks_base AS SELECT * FROM src.chunks WHERE bibcode IN (SELECT unnest(?))", [papers])
    con.execute(
        """CREATE OR REPLACE TABLE subset_role AS
           SELECT unnest(?) AS bibcode, unnest(?) AS role""",
        [papers, ["gold" if b in gold else "neg" if b in neg else "random" for b in papers]],
    )
    n_tok = con.execute("SELECT sum(n_tokens) FROM chunks_base WHERE kind <> 'boilerplate'").fetchone()[0]
    n_ch = con.execute("SELECT count(*) FROM chunks_base WHERE kind <> 'boilerplate'").fetchone()[0]
    print(f"{len(papers)} papers: {len(gold & chunked)} gold, {len(neg & chunked)} hard negatives, {len(rnd)} random; "
          f"{n_ch} embeddable chunks, {n_tok / 1e6:.1f}M tokens")
    print(f"projected embed time: 0.6B ~{n_tok / 3600 / 3600:.1f} h, 4B ~{n_tok / 385 / 3600:.1f} h (chunk text only)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--neg-dense", type=int, default=5)
    s.add_argument("--neg-bm25", type=int, default=3)
    s.add_argument("--random", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--subset-items", action="store_true", help="only the gold/negatives of the Phase 1 item subset")
    args = ap.parse_args()
    if args.cmd == "select":
        select(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

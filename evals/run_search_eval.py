"""
Recall check for `search` against a small gold set (evals/search_gold.json).

For each question it compares, on the same candidate lists:
  semantic    top-k passages from semantic search alone (the old behaviour)
  fused       top-k passages with the --term ranking fused in
  +listing    fused papers plus the first N "other papers matching the term"
  term-only   every paper whose text matches the term (what `grep` would list)
and reports which gold papers each finds. Read-only; needs the `search` pixi env:

    .pixi/envs/search/bin/python -m evals.run_search_eval [--db PATH] [--k 8] [--list 25] [--case ID]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from src import search as S


def _papers(hits: list[S.Hit]) -> list[str]:
    return list(dict.fromkeys(h.bibcode for h in hits))


def run_case(con, model_id: str, size: str, case: dict, k: int, per_paper: int, n_list: int, tw: float | None) -> dict:
    qvecs = S._embed_queries(case["queries"], size)
    n_cand = max(100, 10 * k)
    since = case.get("since")
    ranked = [S._top_chunks(con, model_id, qv, n_cand, since) for qv in qvecs]
    sem = S._fuse(ranked, k, per_paper)
    out = {"semantic": _papers(sem), "fused": _papers(sem), "listing": [], "term_only": []}
    terms = case.get("terms") or []
    if terms:
        pattern, _ = S._compile_terms(terms, False)
        scan = S._term_scan(con, model_id, qvecs, pattern, False, since, n_cand)
        fused = S._fuse(ranked, k, per_paper, scan.ranked, tw)
        out["fused"] = _papers(fused)
        extra = sorted(((b, n) for b, n in scan.per_paper.items() if b not in out["fused"]), key=lambda t: (-t[1], t[0]))
        out["listing"] = [b for b, _ in extra[:n_list]]
        out["term_only"] = list(scan.per_paper)
        out["n_term_chunks"], out["n_term_papers"] = scan.n_chunks, len(scan.per_paper)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--gold", type=Path, default=Path(__file__).parent / "search_gold.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--per-paper", type=int, default=2)
    ap.add_argument("--list", type=int, default=S.DEFAULT_LIST_PAPERS, help="size of the 'other papers' listing")
    ap.add_argument("--model-size", default=config.EMBED_MODEL_SIZE)
    ap.add_argument("--case", default=None, help="run only this case id")
    ap.add_argument("--term-weight", type=float, default=None, help="RRF weight of the term rankings (default: search.TERM_WEIGHT)")
    args = ap.parse_args()

    con = S._connect(args.db)
    model_id = f"Qwen3-Embedding-{args.model_size}"
    cases = [c for c in json.loads(args.gold.read_text())["cases"] if not args.case or c["id"] == args.case]
    totals = {m: [0, 0] for m in ("semantic", "fused", "+listing", "term-only")}
    for case in cases:
        gold = set(case["gold"])
        r = run_case(con, model_id, args.model_size, case, args.k, args.per_paper, args.list, args.term_weight)
        found = {
            "semantic": gold & set(r["semantic"]),
            "fused": gold & set(r["fused"]),
            "+listing": gold & (set(r["fused"]) | set(r["listing"])),
            "term-only": gold & set(r["term_only"]),
        }
        print(f"\n## {case['id']} [{case.get('status', '?')}]  gold={len(gold)}"
              + (f"  term matched {r['n_term_chunks']} chunks / {r['n_term_papers']} papers" if "n_term_chunks" in r else "  (no term)"))
        for mode, got in found.items():
            if mode in ("+listing", "term-only") and not case.get("terms"):
                continue
            print(f"  {mode:10s} {len(got)}/{len(gold)}" + (f"   missing: {sorted(gold - got)}" if gold - got else ""))
            if case.get("terms"):
                totals[mode][0] += len(got)
                totals[mode][1] += len(gold)
        if not case.get("terms") and r["semantic"] != r["fused"]:
            print("  !! no-term case differs between semantic and fused")
    print("\n## totals over the cases that have a --term (gold papers found)")
    for mode, (g, n) in totals.items():
        print(f"  {mode:10s} {g}/{n}" if n else f"  {mode:10s} -")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

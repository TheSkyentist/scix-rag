"""
Pool candidate papers for the Set C questions (evals/sets/setC_questions.json), to be
judged by Claude and confirmed by the user before they become gold.

Per question: the top --dense papers from RRF over the query variants, plus the top
--term papers among those whose (non-boilerplate) chunks match the question's regex,
ranked by their best term-matching chunk's cosine, plus the top --term-count term-matching
papers by matching-chunk count (less tied to the embedder). Each pooled paper carries its
--evidence best chunks (by max cosine over the variants; a term-matching chunk is
always included), clipped to --chars. Read-only on the DB:

    .pixi/envs/embed/bin/python -m evals.pool_setc --out DIR [--dense 30] [--term 30]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from evals.run_retrieval_eval import DEPTH, SETS_DIR, Ctx, _rrf
from src import search as S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dense", type=int, default=30)
    ap.add_argument("--term", type=int, default=30)
    ap.add_argument("--term-count", type=int, default=15, help="also the top N term-matching papers by matching-chunk count")
    ap.add_argument("--evidence", type=int, default=2)
    ap.add_argument("--chars", type=int, default=800)
    args = ap.parse_args()

    qs = json.loads((SETS_DIR / "setC_questions.json").read_text())["questions"]
    con = S._connect(args.db)
    ctx = Ctx(con, config.EMBED_MODEL)
    ix = ctx.index
    rows_of: dict[str, np.ndarray] = {}
    order = np.argsort(ix.bibcode, kind="stable")
    bs = ix.bibcode[order]
    cuts = np.flatnonzero(bs[1:] != bs[:-1]) + 1
    for grp in np.split(order, cuts):
        rows_of[ix.bibcode[grp[0]]] = grp
    meta = {b: (fa, y, t) for b, fa, y, t in con.execute("SELECT bibcode, first_author, year, title FROM papers").fetchall()}
    args.out.mkdir(parents=True, exist_ok=True)

    for q in qs:
        qv = np.stack(ctx.qvecs(q["queries"]))
        ranked = _rrf([ix.top(v, DEPTH, q["since"]) for v in qv])
        dense_papers = list(dict.fromkeys(b for b, _ in ranked))[: args.dense]
        pattern, _ = S._compile_terms(q["terms"], False)
        scan = S._term_scan(con, ctx.model_id, [v.tolist() for v in qv], pattern, False, q["since"], 10**6)
        best_term: dict[str, tuple[float, int]] = {}
        for lst in scan.ranked:
            for b, i, s in lst:
                if b not in best_term or s > best_term[b][0]:
                    best_term[b] = (s, i)
        term_papers = sorted(best_term, key=lambda b: -best_term[b][0])[: args.term]
        by_count = [b for b, _ in sorted(scan.per_paper.items(), key=lambda t: (-t[1], t[0]))[: args.term_count]]
        pool = list(dict.fromkeys(dense_papers + term_papers + by_count))

        recs = []
        for b in pool:
            rows = rows_of[b]
            sc = (ix.M[rows] @ qv.T).max(axis=1)
            pick = [int(ix.chunk_index[rows[j]]) for j in np.argsort(-sc)[: args.evidence]]
            if b in best_term and best_term[b][1] not in pick:
                pick = pick[: args.evidence - 1] + [best_term[b][1]]
            ev = []
            for i in pick:
                hp, text = con.execute("SELECT heading_path, text FROM chunks WHERE bibcode = ? AND chunk_index = ?", [b, i]).fetchone()
                text = re.sub(r"\s+", " ", text)
                ev.append({"chunk": i, "heading": hp, "text": text[: args.chars] + (" …" if len(text) > args.chars else "")})
            fa, y, t = meta[b]
            recs.append({"bibcode": b, "first_author": fa, "year": y, "title": t,
                         "via": [v for v, ok in (("dense", b in dense_papers), ("term", b in term_papers), ("term-count", b in by_count)) if ok],
                         "n_term_chunks": scan.per_paper.get(b, 0), "evidence": ev})
        (args.out / f"{q['id']}.json").write_text(json.dumps(
            {"id": q["id"], "question": q["queries"][0], "pool": recs}, ensure_ascii=False, indent=1))
        print(f"{q['id']}: {len(pool)} pooled ({len(dense_papers)} dense, {len(term_papers)} term of "
              f"{len(scan.per_paper)} term-matching papers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

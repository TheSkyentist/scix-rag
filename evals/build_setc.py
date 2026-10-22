"""
Merge Claude's relevance judgments for the Set C pools into evals/sets/setC.jsonl and
write evals/sets/setC_review.md (the gold lists, for the user to confirm or correct).

    .pixi/envs/embed/bin/python -m evals.build_setc --pools DIR

DIR holds <id>.json (pools from evals.pool_setc) and <id>.judged.json. User corrections
go in evals/sets/setC_overrides.json as {"<id>": {"<bibcode>": rel, ...}} and win over
the judgments (rel 0 removes a paper); rebuild after editing it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

import config
from evals.run_retrieval_eval import SETS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--pools", type=Path, required=True)
    args = ap.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    meta = {b: (fa, y, t) for b, fa, y, t in con.execute("SELECT bibcode, first_author, year, title FROM papers").fetchall()}
    overrides_file = SETS_DIR / "setC_overrides.json"
    overrides = json.loads(overrides_file.read_text()) if overrides_file.exists() else {}
    qs = json.loads((SETS_DIR / "setC_questions.json").read_text())["questions"]

    items, md = [], ["# Set C gold lists — please confirm\n",
                     "Graded by Claude from pooled candidates (dense search + regex matches + its own searches). "
                     "**2** = substantially answers, **1** = relevant/partial. To correct, add entries to "
                     "`evals/sets/setC_overrides.json` (`{\"C-xray\": {\"BIBCODE\": 2}}`; 0 removes) or just tell me.\n"]
    for q in qs:
        jf = args.pools / f"{q['id']}.judged.json"
        if not jf.exists():
            print(f"skip {q['id']}: no judgments")
            continue
        j = json.loads(jf.read_text())
        pool = {r["bibcode"] for r in json.loads((args.pools / f"{q['id']}.json").read_text())["pool"]}
        rel, why = {}, {}
        for x in j["judgments"] + j.get("added", []):
            if x["bibcode"] not in meta:
                print(f"{q['id']}: unknown bibcode {x['bibcode']} dropped")
                continue
            rel[x["bibcode"]], why[x["bibcode"]] = int(x["rel"]), x.get("why", "")
        missing = pool - set(rel)
        if missing:
            print(f"{q['id']}: {len(missing)} pooled papers unjudged")
        for b, r in overrides.get(q["id"], {}).items():
            rel[b], why[b] = int(r), "user override"
        items.append({"id": q["id"], "set": "C", "queries": q["queries"], "since": q.get("since"),
                      "terms": q.get("terms", []), "rel": rel})
        n2, n1 = sum(r == 2 for r in rel.values()), sum(r == 1 for r in rel.values())
        added = {x["bibcode"] for x in j.get("added", [])}
        md.append(f"\n## {q['id']}: {q['queries'][0]}\n\n{n2} × rel 2, {n1} × rel 1 "
                  f"({len(added)} added by the judge beyond the pool)\n")
        for r in (2, 1):
            for b in sorted((b for b, v in rel.items() if v == r), key=lambda b: (-(meta[b][1] or 0), b)):
                fa, y, t = meta[b]
                md.append(f"- **{r}** {b} — {(fa or '?').split(',')[0]} {y}: {t}"
                          + (" *(added)*" if b in added else "") + f"  \n  _{why[b]}_")
    (SETS_DIR / "setC.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
    (SETS_DIR / "setC_review.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(items)} Set C items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

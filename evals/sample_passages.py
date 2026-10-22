"""
Sample chunks for eval Set B (passage-level gold). Claude later writes one research
question per chunk that the chunk answers; the chunk (and its paper) is the gold.

Stratified by publication era and chunk kind, one chunk per paper, skipping very
short chunks and introductions (whose content is mostly restated from other papers,
so a single-chunk gold would be unfair). Read-only on the DB:

    .pixi/envs/embed/bin/python -m evals.sample_passages --out PATH [--n 300] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

import config

# (label, year filter, share of the sample)
_ERAS = [("<=2015", "p.year <= 2015", 0.2), ("2016-2022", "p.year BETWEEN 2016 AND 2022", 0.3), (">=2023", "p.year >= 2023", 0.5)]
_TABLE_SHARE = 0.1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    con.execute(f"SELECT setseed({(args.seed % 1000) / 1000})")
    out = []
    for era, where, share in _ERAS:
        for kind, kshare in (("prose", 1 - _TABLE_SHARE), ("table", _TABLE_SHARE)):
            n = round(args.n * share * kshare)
            rows = con.execute(
                f"""WITH c AS (
                      SELECT c.bibcode, c.chunk_index, c.heading_path, c.text, c.n_tokens, p.title, p.year,
                             row_number() OVER (PARTITION BY c.bibcode ORDER BY random()) AS r
                      FROM chunks c JOIN papers p USING (bibcode)
                      WHERE c.kind = ? AND {where} AND c.n_tokens >= 150
                        AND NOT regexp_matches(coalesce(c.heading_path, ''), '(?i)^\\W*(\\d+\\.?\\s*)?introduction')
                        AND (? = 'table' OR c.chunk_index > 0))
                    SELECT bibcode, chunk_index, heading_path, text, n_tokens, title, year
                    FROM c WHERE r = 1 ORDER BY random() LIMIT ?""",
                [kind, kind, n],
            ).fetchall()
            for b, i, hp, t, nt, title, y in rows:
                out.append({"bibcode": b, "chunk_index": i, "kind": kind, "era": era, "year": y,
                            "title": title, "heading_path": hp, "n_tokens": nt, "text": t})
    seen, uniq = set(), []
    for c in out:  # one chunk per paper across strata too
        if c["bibcode"] not in seen:
            seen.add(c["bibcode"])
            uniq.append(c)
    for i, c in enumerate(uniq):
        c["id"] = f"B{i:04d}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in uniq))
    print(f"wrote {len(uniq)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

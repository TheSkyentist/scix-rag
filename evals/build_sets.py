"""
Assemble eval Sets A and B from the mined candidates plus Claude-written queries.

    .pixi/envs/embed/bin/python -m evals.build_sets --work DIR

DIR holds the candidate batches (A_*.jsonl, B_*.jsonl) and Claude's answers
(A_*.out.jsonl, B_*.out.jsonl). Kept queries get a lexical-overlap score against
their gold (Set A: the cited paper's title + abstract; Set B: the gold chunk) and a
bucket, so results can be split by how much a query leans on shared wording.
Set A queries that restate the cited title (>= TITLE_OVERLAP_MAX of its content
words) are dropped (LitSearch filter); the rest carry a title-overlap tag.
Writes evals/sets/setA.jsonl and evals/sets/setB.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb

import config

SETS_DIR = Path(__file__).parent / "sets"
TITLE_OVERLAP_MAX = 0.75  # drop Set A queries containing >= this share of the cited title's content words

_STOP = set("""
a an the and or of in on at to for from by with without as is are was were be been being this that these those
it its into than then there their they them which what who whom whose how why when where does do did can could
would should may might will shall not no nor any some such about over under between within across via per also
using use used based both each other more most less very much many few here we our you your i e g et al vs
""".split())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3 and w not in _STOP}


def _overlap(query: str, target: str) -> float:
    q = _words(query)
    return len(q & _words(target)) / len(q) if q else 0.0


def _bucket(x: float) -> str:
    return "low" if x < 0.34 else "mid" if x < 0.6 else "high"


def _load(work: Path, prefix: str) -> list[tuple[dict, dict]]:
    pairs = []
    for cand_file in sorted(work.glob(f"{prefix}_*.jsonl")):
        if cand_file.name.endswith(".out.jsonl"):
            continue
        out_file = cand_file.with_name(cand_file.stem + ".out.jsonl")
        cands = {c["id"]: c for c in map(json.loads, cand_file.read_text().splitlines())}
        outs = {o["id"]: o for o in map(json.loads, out_file.read_text().splitlines()) if o}
        missing = set(cands) - set(outs)
        if missing:
            print(f"warning: {out_file.name} lacks {len(missing)} ids")
        pairs += [(cands[i], outs[i]) for i in cands if i in outs]
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    con = duckdb.connect(str(args.db), read_only=True)
    SETS_DIR.mkdir(exist_ok=True)

    a_pairs = _load(args.work, "A")
    meta = dict(con.execute(
        "SELECT bibcode, [title, coalesce(abstract, '')] FROM papers WHERE bibcode IN (SELECT unnest(?))",
        [[c["cited"] for c, _ in a_pairs]],
    ).fetchall())
    set_a, n_title = [], 0
    for c, o in a_pairs:
        if not o.get("keep") or not o.get("query"):
            continue
        title, abstract = meta[c["cited"]]
        tw = _words(title)
        t_ov = len(_words(o["query"]) & tw) / len(tw) if tw else 0.0
        if t_ov >= TITLE_OVERLAP_MAX:
            n_title += 1
            continue
        ov = _overlap(o["query"], f"{title} {abstract}")
        set_a.append({
            "id": c["id"], "set": "A", "queries": [o["query"]], "gold_papers": [c["cited"]],
            "exclude": [c["citing"]], "source_sentence": c["sentence"],
            "tags": {"specificity": o.get("specificity"), "overlap": round(ov, 3), "bucket": _bucket(ov),
                     "title_overlap": round(t_ov, 3), "title_bucket": _bucket(t_ov),
                     "cited_year": c["cited_year"]},
        })
    print(f"Set A: {len(a_pairs)} candidates -> {sum(1 for _, o in a_pairs if o.get('keep'))} kept by Claude "
          f"-> {len(set_a)} after title-overlap filter ({n_title} dropped)")

    set_b = []
    b_pairs = _load(args.work, "B")
    for c, o in b_pairs:
        if not o.get("keep") or not o.get("query"):
            continue
        ov = _overlap(o["query"], c["text"])
        set_b.append({
            "id": c["id"], "set": "B", "queries": [o["query"]], "gold_papers": [c["bibcode"]],
            "gold_chunks": [[c["bibcode"], c["chunk_index"]]], "exclude": [],
            "tags": {"qtype": o.get("qtype"), "kind": c["kind"], "era": c["era"], "overlap": round(ov, 3),
                     "bucket": _bucket(ov)},
        })
    print(f"Set B: {len(b_pairs)} candidates -> {len(set_b)} kept")

    for name, items in (("setA", set_a), ("setB", set_b)):
        (SETS_DIR / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

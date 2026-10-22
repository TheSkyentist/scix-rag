"""
Mine citation-context sentences for eval Set A (LitSearch-style gold).

A citing paper's prose sentence that cites exactly one other in-corpus paper by
author-year ("Smith et al. 2023", "Smith & Jones (2023)", "Smith 2023") becomes a
candidate: Claude later rewrites the sentence as a research question and the cited
paper is the gold answer. Citations are resolved against the citing paper's own
`citation_edges` (surname + year, year-1 allowed for preprint years), and only
unambiguous matches are kept. Read-only on the DB:

    .pixi/envs/embed/bin/python -m evals.mine_citation_contexts --out PATH [--n 700] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb

import config

# A sentence ends at ". " + capital, but not after common abbreviations.
_ABBREV = r"(?<!\bal)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bFigs)(?<!\bEq)(?<!\bEqs)(?<!\bSect)(?<!\bSec)(?<!\bvs)(?<!\bcf)(?<!\bapprox)(?<!\bTab)(?<!\bRef)(?<!\bResp)(?<!\bresp)"
_SENT_END = re.compile(_ABBREV + r"[.!?]\s+(?=[A-Z(\[])")
# Anything that looks like an author-year citation; used to require exactly one per sentence.
_ANY_CITE = re.compile(r"[A-Z][\w'’\-]+(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][\w'’\-]+)?,?\s*\(?\s*(?:19|20)\d\d[a-z]?\b")
_YEAR = re.compile(r"\b(?:19|20)\d\d[a-z]?\b")


def _fold(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def _surname(first_author: str | None) -> str | None:
    if not first_author:
        return None
    s = first_author.split(",")[0].strip()
    return s if len(s) >= 2 else None


def _cite_rx(surname: str, year: int) -> re.Pattern:
    sn = re.escape(_fold(surname)).replace(r"\ ", r"\s+")
    yrs = f"(?:{year}|{year - 1})"
    return re.compile(
        rf"\b{sn}(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][\w'’\-]+)?,?\s*\(?\s*{yrs}[a-z]?\b"
    )


def _sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        spans.append((start, m.start() + 1))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=700, help="candidates to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    chunked = {b for (b,) in con.execute("SELECT DISTINCT bibcode FROM chunks").fetchall()}
    meta = {
        b: (fa, y, t) for b, fa, y, t in
        con.execute("SELECT bibcode, first_author, year, title FROM papers").fetchall()
    }
    cites: dict[str, list[str]] = defaultdict(list)
    for a, b in con.execute("SELECT citing_bibcode, cited_bibcode FROM citation_edges").fetchall():
        if a in chunked and b in chunked and a != b:
            cites[a].append(b)

    # Per citing paper, the cited in-corpus papers as (bibcode, regex), dropping
    # (surname, year) keys shared by two cited papers -- those can't be resolved.
    targets: dict[str, list[tuple[str, re.Pattern]]] = {}
    for a, bs in cites.items():
        keys = defaultdict(list)
        for b in bs:
            fa, y, _ = meta.get(b, (None, None, None))
            sn = _surname(fa)
            if sn and y:
                keys[(_fold(sn).lower(), y)].append(b)
        targets[a] = [
            (bs_[0], _cite_rx(meta[bs_[0]][0].split(",")[0].strip(), y))
            for (sn, y), bs_ in keys.items() if len(bs_) == 1
        ]

    cands = []
    rows = con.execute(
        "SELECT bibcode, chunk_index, heading_path, text FROM chunks WHERE kind = 'prose' ORDER BY bibcode, chunk_index"
    ).fetchall()
    for bibcode, idx, hp, text in rows:
        tl = targets.get(bibcode)
        if not tl:
            continue
        ftext = _fold(text)  # match accent-folded text (surnames are folded too)
        for s, e in _sentences(ftext):
            sent = ftext[s:e].strip()
            if len(_ANY_CITE.findall(sent)) != 1 or len(_YEAR.findall(sent)) != 1:
                continue
            hits = [b for b, rx in tl if rx.search(sent)]
            if len(hits) != 1:
                continue
            words = _ANY_CITE.sub("", sent).split()
            if not 14 <= len(words) <= 70 or "$" in sent and sent.count("$") > 6:
                continue
            cands.append({
                "citing": bibcode, "chunk_index": idx, "heading_path": hp,
                "sentence": sent, "cited": hits[0],
            })

    print(f"{len(cands)} candidate sentences from {len({c['citing'] for c in cands})} citing papers "
          f"-> {len({c['cited'] for c in cands})} distinct cited papers")
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    # Spread over targets and citing papers: <=1 sentence per cited paper, <=3 per citing paper.
    out, per_cited, per_citing = [], set(), defaultdict(int)
    for c in cands:
        if c["cited"] in per_cited or per_citing[c["citing"]] >= 3:
            continue
        per_cited.add(c["cited"])
        per_citing[c["citing"]] += 1
        fa, y, t = meta[c["cited"]]
        c["cited_year"], c["cited_title"] = y, t
        c["citing_year"] = meta[c["citing"]][1]
        out.append(c)
        if len(out) >= args.n:
            break
    for i, c in enumerate(out):
        c["id"] = f"A{i:04d}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out))
    print(f"wrote {len(out)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

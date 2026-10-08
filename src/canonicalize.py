"""
Keep one row per paper, keyed on ADS's canonical bibcode.

ADS re-keys records over time: a preprint (`2026arXiv260104955C`) becomes the
journal record (`2026Natur.657..621C`) and the old bibcode turns into an
`alternate_bibcode`; a journal record gains an arXiv identifier. ads_query only
ever inserts bibcodes it hasn't seen, so the DB drifts: twin rows for one paper,
stale `arxiv_id`/`esources`, edges pointing at retired bibcodes.

Given fresh ADS docs, this groups the DB rows that are the same paper (same
canonical bibcode after resolving aliases, or the same arXiv id) and, per group:
  - keeps/creates one `papers` row under the canonical bibcode (earliest
    `query_source`/`added_at` retained), with `arxiv_id`, `alternate_bibcode`
    and `esources` refreshed from ADS;
  - repoints `citation_edges` (also endpoints that are merely aliases of a known
    paper), dropping duplicates and self-loops;
  - keeps ONE member's chunks/embeddings (preferring chunks built from the final
    arXiv stem, then the largest) and re-keys them to the canonical bibcode --
    no re-embedding; `chunk` re-chunks it later if its source stem is outdated;
  - moves/deletes raw files named by the old bibcode stem;
  - rewrites stems in `fetch_ignore.txt`/`convert_ignore.txt` (a paper that now
    has an arxiv_id has its ignore line dropped so it is retried via arXiv) and
    bibcodes in evals/search_gold.json.

Rows whose bibcode is already canonical and alone in their group only get
`arxiv_id`/`alternate_bibcode`/`esources` refreshed.

Run directly (ADS lookup of every paper, ~150 requests):
    pixi run canonicalize             # dry run: print what would change
    pixi run canonicalize --apply
    pixi run canonicalize --since 2008 [--apply]
`fetch --refresh` and ads_query call `run()` with docs they already have.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import duckdb

import config

from .ads_query import (
    _PAPER_COLUMNS,
    _UPSERT_PAPERS,
    _doc_to_params,
    _extract_arxiv_id,
    _fetch_many_by_bibcode,
    _session,
)

_GOLD_PATH = config.ROOT / "evals" / "search_gold.json"
_PROVISIONAL_BIBCODE_RE = re.compile(r"^\d{4}(arXiv|.{5}\.tmp)")  # preprint / in-press placeholder
_RAW_DIRS = ((config.MARKDOWN_DIR, "md"), (config.HTML_DIR, "html"), (config.PDF_DIR, "pdf"))


def _raw_files(stem: str) -> list[Path]:
    return [d / f"{stem}.{ext}" for d, ext in _RAW_DIRS if (d / f"{stem}.{ext}").exists()]


def _done(stem: str) -> bool:
    return bool(_raw_files(stem))


def _pick_target(canonicals: set[str]) -> str:
    final = sorted(c for c in canonicals if not _PROVISIONAL_BIBCODE_RE.match(c))
    return (final or sorted(canonicals))[0]


def _plan(con: duckdb.DuckDBPyConnection, docs: list[dict]) -> dict:
    canon_of: dict[str, str] = {}  # any known bibcode (canonical or alias) -> canonical
    doc_of: dict[str, dict] = {}
    for d in docs:
        c = d.get("bibcode")
        if not c:
            continue
        doc_of[c] = d
        for a in [c, *(d.get("alternate_bibcode") or [])]:
            canon_of.setdefault(a, c)

    rows = {
        b: (arxiv, added, src)
        for b, arxiv, added, src in con.execute(
            "SELECT bibcode, arxiv_id, added_at, query_source FROM papers"
        ).fetchall()
    }

    def ads_arxiv(b: str) -> str | None:
        d = doc_of.get(canon_of.get(b, ""))
        return (_extract_arxiv_id(d.get("identifier")) if d else None) or rows[b][0]

    parent = {b: b for b in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_by: dict[tuple, str] = {}
    for b in rows:
        keys = [("c", canon_of.get(b, b))]
        if (a := ads_arxiv(b)):
            keys.append(("a", a))
        for key in keys:
            if key in first_by:
                parent[find(b)] = find(first_by[key])
            else:
                first_by[key] = b

    groups: dict[str, list[str]] = defaultdict(list)
    for b in rows:
        groups[find(b)].append(b)

    refresh, merges, disagreements = [], [], []
    for members in groups.values():
        canonicals = {canon_of[m] for m in members if m in canon_of}
        if not canonicals:
            continue  # nothing from ADS for this group; out of scope
        target = _pick_target(canonicals)
        arxiv = ads_arxiv(target) if target in rows else None
        arxiv = arxiv or next((a for m in members if (a := ads_arxiv(m))), None)
        entry = {"members": sorted(members), "target": target, "arxiv_id": arxiv}
        if len(canonicals) > 1:
            disagreements.append(entry)
        if members == [target]:
            refresh.append(entry)
        else:
            merges.append(entry)

    removed = {m for e in merges for m in e["members"] if m != e["target"]}
    final_papers = (set(rows) - removed) | {e["target"] for e in merges}
    remap = {m: e["target"] for e in merges for m in e["members"] if m != e["target"]}
    dangling = {  # edge endpoints that are not a paper row
        r[0]
        for r in con.execute(
            """SELECT b FROM (SELECT citing_bibcode b FROM citation_edges
                              UNION SELECT cited_bibcode FROM citation_edges)
               WHERE b NOT IN (SELECT bibcode FROM papers)"""
        ).fetchall()
    }
    n_retired = len(remap)
    for alias in dangling:  # retired bibcodes that only survive as edge endpoints
        c = canon_of.get(alias)
        if c and c != alias and c in final_papers:
            remap[alias] = c
    return {
        "rows": rows, "canon_of": canon_of, "doc_of": doc_of, "refresh": refresh,
        "merges": merges, "disagreements": disagreements, "remap": remap,
        "n_retired": n_retired, "n_dangling": len(dangling),
    }


def _chunk_info(con: duckdb.DuckDBPyConnection, bibcodes: list[str]) -> dict[str, tuple[int, str | None]]:
    return {
        b: (n, a)
        for b, n, a in con.execute(
            "SELECT bibcode, count(*), any_value(arxiv_id) FROM chunks WHERE bibcode = ANY(?) GROUP BY bibcode",
            [bibcodes],
        ).fetchall()
    }


def _donor(members: list[str], arxiv: str | None, info: dict) -> str | None:
    with_chunks = [m for m in members if m in info]
    if not with_chunks:
        return None
    # chunks built from the final arXiv stem first, then the biggest set
    return max(with_chunks, key=lambda m: (bool(arxiv) and info[m][1] == arxiv, info[m][0]))


def _rewrite_lines(path: Path, mapping: dict[str, str | None], strip_ext: bool) -> int:
    """Rename (or drop, when mapped to None) stems in an ignore list; keep comments."""
    if not path.exists() or not mapping:
        return 0
    changed, out = 0, []
    lines = path.read_text(encoding="utf-8").splitlines()
    present = {name for ln in lines if (name := ln.split("#", 1)[0].strip())}
    for line in lines:
        name = line.split("#", 1)[0].strip()
        ext = ""
        stem = name
        if strip_ext and "." in name and name.rsplit(".", 1)[1] in ("pdf", "html", "md"):
            stem, ext = name.rsplit(".", 1)[0], "." + name.rsplit(".", 1)[1]
        if name and stem in mapping:
            changed += 1
            new = mapping[stem]
            if new is None or new + ext in present:  # dropped, or the new name is already listed
                continue
            present.add(new + ext)
            line = line.replace(name, new + ext, 1)
        out.append(line)
    if changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def _settle_files(old: str, target: str, arxiv: str | None) -> None:
    """Files named by an old bibcode stem: drop if the paper's real stem already has content,
    else rename to the canonical bibcode."""
    for f in _raw_files(old):
        dest = f.with_name(f"{target}{f.suffix}")
        if (arxiv and _done(arxiv)) or dest.exists():
            f.unlink()
            print(f"  [drop] {f.relative_to(config.ROOT)}")
        else:
            f.rename(dest)
            print(f"  [move] {f.relative_to(config.ROOT)} -> {dest.name}")


def run(con: duckdb.DuckDBPyConnection, docs: list[dict], apply: bool, verbose: int = 40) -> dict:
    p = _plan(con, docs)
    rows, doc_of, remap = p["rows"], p["doc_of"], p["remap"]

    gained = [
        e for e in p["refresh"]
        if e["arxiv_id"] and not rows[e["target"]][0]
    ]
    all_members = [m for e in p["merges"] for m in e["members"]]
    info = _chunk_info(con, all_members) if all_members else {}

    con.execute("CREATE OR REPLACE TEMP TABLE remap(old_b VARCHAR, new_b VARCHAR)")
    if remap:
        con.executemany("INSERT INTO remap VALUES (?, ?)", list(remap.items()))
    edge_stats = con.execute(
        """SELECT count(*) FILTER (WHERE m1.old_b IS NOT NULL OR m2.old_b IS NOT NULL)
           FROM citation_edges e
           LEFT JOIN remap m1 ON e.citing_bibcode = m1.old_b
           LEFT JOIN remap m2 ON e.cited_bibcode = m2.old_b"""
    ).fetchone()[0]

    print(f"{len(p['refresh']) + len(p['merges'])} identity groups examined from {len(docs)} ADS records")
    print(f"  {len(gained)} existing rows gained an arxiv_id")
    print(f"  {len(p['merges'])} groups to merge/rename ({p['n_retired']} bibcodes retired)")
    print(f"  {len(remap) - p['n_retired']} of {p['n_dangling']} dangling edge endpoints resolve to a known paper")
    print(f"  {edge_stats} edges to repoint")
    for e in p["merges"][:verbose]:
        donor = _donor(e["members"], e["arxiv_id"], info)
        extra = [m for m in e["members"] if m != e["target"]]
        print(f"  {' + '.join(extra) or '(rename)'} -> {e['target']}"
              f"  arxiv={e['arxiv_id']} chunks_from={donor}")
    if len(p["merges"]) > verbose:
        print(f"  ... and {len(p['merges']) - verbose} more")
    for e in gained[:verbose]:
        print(f"  gained arxiv_id: {e['target']} -> {e['arxiv_id']}")
    for e in p["disagreements"]:
        print(f"  [warn] ADS gives several canonical bibcodes for {e['members']}; using {e['target']}")

    if not apply:
        print("Dry run: nothing written")
        return {"gained": len(gained), "merged": len(p["merges"])}

    # Edges first (needs the remap table), then papers/chunks, all in one transaction.
    con.begin()
    try:
        for e in p["refresh"]:
            d = doc_of[e["target"]]
            con.execute(
                "UPDATE papers SET arxiv_id = COALESCE(?, arxiv_id), alternate_bibcode = ?, esources = ? "
                "WHERE bibcode = ?",
                [e["arxiv_id"], d.get("alternate_bibcode") or [], d.get("esources") or [], e["target"]],
            )
        file_jobs, ignore_fetch, ignore_convert, gold = [], {}, {}, {}
        for e in p["merges"]:
            target, members, arxiv = e["target"], e["members"], e["arxiv_id"]
            earliest = min(members, key=lambda m: rows[m][1] or "")
            params = dict(zip(_PAPER_COLUMNS, _doc_to_params(doc_of[target], rows[earliest][2])))
            params["arxiv_id"] = arxiv
            con.execute(_UPSERT_PAPERS, [params[c] for c in _PAPER_COLUMNS])
            con.execute("UPDATE papers SET esources = ? WHERE bibcode = ?",
                        [doc_of[target].get("esources") or [], target])
            old = [m for m in members if m != target]
            con.execute("DELETE FROM papers WHERE bibcode = ANY(?)", [old])

            donor = _donor(members, arxiv, info)
            for table in ("chunk_embeddings", "chunks"):
                drop = [m for m in members if m != donor]
                con.execute(f"DELETE FROM {table} WHERE bibcode = ANY(?)", [drop])
            if donor and donor != target:
                con.execute(
                    "INSERT INTO chunks (bibcode, arxiv_id, chunk_index, section_heading, heading_path, kind, text, n_tokens) "
                    "SELECT ?, arxiv_id, chunk_index, section_heading, heading_path, kind, text, n_tokens "
                    "FROM chunks WHERE bibcode = ?", [target, donor])
                con.execute(
                    "INSERT INTO chunk_embeddings (bibcode, chunk_index, model, text_hash, embedding) "
                    "SELECT ?, chunk_index, model, text_hash, embedding FROM chunk_embeddings WHERE bibcode = ?",
                    [target, donor])
                con.execute("DELETE FROM chunk_embeddings WHERE bibcode = ?", [donor])
                con.execute("DELETE FROM chunks WHERE bibcode = ?", [donor])
            file_jobs.append((donor, old, target, arxiv))
            for m in old:
                gold[m] = target
                ignore_fetch[m] = None if arxiv else target
                ignore_convert[m] = None if arxiv else target
            if arxiv:
                ignore_fetch[target] = None  # stale bibcode-stem line; retried via arXiv
                ignore_convert[target] = None
        for e in gained:
            ignore_fetch[e["target"]] = None
            ignore_convert[e["target"]] = None

        con.execute(
            """CREATE OR REPLACE TEMP TABLE new_edges AS
               SELECT DISTINCT COALESCE(m1.new_b, e.citing_bibcode) AS c, COALESCE(m2.new_b, e.cited_bibcode) AS d
               FROM citation_edges e
               LEFT JOIN remap m1 ON e.citing_bibcode = m1.old_b
               LEFT JOIN remap m2 ON e.cited_bibcode = m2.old_b
               WHERE m1.old_b IS NOT NULL OR m2.old_b IS NOT NULL"""
        )
        con.execute("DELETE FROM citation_edges WHERE citing_bibcode IN (SELECT old_b FROM remap) "
                    "OR cited_bibcode IN (SELECT old_b FROM remap)")
        con.execute("INSERT INTO citation_edges SELECT c, d FROM new_edges WHERE c <> d ON CONFLICT DO NOTHING")
        con.commit()
    except Exception:
        con.rollback()
        raise

    # Everything outside the DB, after the commit.
    for donor, old, target, arxiv in file_jobs:
        for m in ([donor] if donor and donor != target and donor in old else []) + [m for m in old if m != donor]:
            _settle_files(m, target, arxiv)
    # the donor's files move first (above), so a stem that is the donor's bibcode keeps its content
    fcount = _rewrite_lines(config.FETCH_IGNORE_PATH, ignore_fetch, strip_ext=False)
    ccount = _rewrite_lines(config.CONVERT_IGNORE_PATH, ignore_convert, strip_ext=True)
    if _GOLD_PATH.exists() and gold:
        text, n = _GOLD_PATH.read_text(encoding="utf-8"), 0
        for o, t in gold.items():
            n += text.count(f'"{o}"')
            text = text.replace(f'"{o}"', f'"{t}"')
        if n:
            _GOLD_PATH.write_text(text, encoding="utf-8")
        print(f"  evals/search_gold.json: {n} bibcodes updated")
    print(f"  ignore lists: {fcount} fetch / {ccount} convert lines renamed or dropped")
    return {"gained": len(gained), "merged": len(p["merges"])}


def canonicalize(apply: bool, since: int | None = None) -> None:
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS esources VARCHAR[]")
    sql, params = "SELECT bibcode FROM papers", []
    if since:
        sql, params = sql + " WHERE year >= ?", [since]
    bibcodes = [r[0] for r in con.execute(sql, params).fetchall()]
    print(f"Looking up {len(bibcodes)} papers on ADS...")
    with _session() as client:
        docs = _fetch_many_by_bibcode(client, bibcodes)
    print(f"  -> {len(docs)} records")
    run(con, docs, apply=apply)
    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge duplicate papers under ADS's canonical bibcode")
    parser.add_argument("--apply", action="store_true", help="Write the changes (default: dry run)")
    parser.add_argument("--since", type=int, default=None, help="Only look up papers from this year on")
    args = parser.parse_args()
    canonicalize(apply=args.apply, since=args.since)

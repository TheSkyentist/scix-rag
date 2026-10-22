"""
Command-line search over the local literature index in papers.duckdb. Retrieval
only: the calling agent (or you) does the reading and writing. Read-only on the
database, and needs neither .env nor ADS access.

Subcommands:
    search  "q1" ["q2" ...]   semantic search (embeds the queries with the same
                              embedding model as the chunks, scores in
                              DuckDB, fuses the query variants by reciprocal rank).
                              With --term REGEX (repeatable) the search is also run
                              restricted to chunks whose text matches the regex, that
                              ranking is fused in too, matches are shown as
                              keyword-in-context snippets, and every other paper that
                              matches is listed after the passages.
    chunk   ID [ID ...]       one passage plus its neighbours (ID = BIBCODE#INDEX)
    paper   BIBCODE           metadata + abstract + outline, or a section/the text
    grep    PATTERN           literal (or --regex) text match over chunk text: the
                              exhaustive, un-ranked "every paper that says X" tool,
                              also for checking exact numbers and quotes
    status                    what is indexed / embedded

Scoring happens inside DuckDB with the core list_inner_product() function on the
stored FLOAT[] vectors (unit length, so inner product == cosine); vectors are
never pulled into Python. Without --term the scoring query touches only
chunk_embeddings; with it, one regex scan of `chunks` picks the matching chunk ids
and their scores for all query variants are computed in a single pass. The big
`chunks` / `papers` tables are otherwise read only for the final hits.
Only `search` imports mlx/transformers, so the other subcommands start fast.

Run via the wrapper (bin/arxivrag-search) or, from the repo:
    pixi run search search "Ca II triplet absorption" --term 'ca\\W*(ii)?\\W*triplet'
Always single-quote bibcodes, queries and regexes: bibcodes contain '&'.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import duckdb

import config

RRF_K = 60
MAX_QUERIES = 4
COVERAGE_WARN = 0.99
GREP_ROW_CAP = 20_000  # bound memory on very common terms
TERM_TOO_GENERIC = 5_000  # a --term matching this many chunks no longer sharpens the ranking
DEFAULT_LIST_PAPERS = 25  # `search`: other term-matching papers listed after the passages
MAX_LIST_PAPERS = 300
TERM_WEIGHT = 1.0  # RRF weight of the term-restricted rankings relative to the semantic ones


class Hit(NamedTuple):
    bibcode: str
    idx: int
    rrf: float
    score: float  # best cosine across the query variants
    queries: tuple[int, ...]  # 1-based variants that retrieved it
    term: bool = False  # chunk text matches --term


class TermScan(NamedTuple):
    ranked: list[list[tuple[str, int, float]]]  # per query variant: top matching chunks by cosine
    per_paper: Counter  # bibcode -> number of matching (non-boilerplate) chunks
    n_chunks: int


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


class _Phases:
    """--timing: wall time per phase, printed to stderr."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.last = time.perf_counter()
        self.marks: list[tuple[str, float]] = []

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        self.marks.append((label, now - self.last))
        self.last = now

    def report(self) -> None:
        if self.enabled:
            print("timing: " + " | ".join(f"{k} {v:.2f}s" for k, v in self.marks), file=sys.stderr)


def _connect(db: Path) -> duckdb.DuckDBPyConnection:
    if not db.exists():
        _die(f"database not found: {db}")
    try:
        return duckdb.connect(str(db), read_only=True)
    except duckdb.IOException as e:
        _die(
            f"cannot open {db} read-only ({str(e).splitlines()[0]}). If `pixi run embed` or another "
            "pipeline stage is running, retry when it finishes.",
            2,
        )


def _models(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    try:
        return dict(con.execute("SELECT model, count(*) FROM chunk_embeddings GROUP BY 1").fetchall())
    except duckdb.CatalogException:
        return {}


def _in_list(n: int) -> str:
    return ",".join("?" * n)


def _author_label(first_author: str | None, n_authors: int | None) -> str:
    if not first_author:
        return "Unknown"
    surname = first_author.split(",")[0].strip()
    return f"{surname} et al." if (n_authors or 0) > 1 else surname


def _clip(text: str, max_chars: int) -> tuple[str, int]:
    """Cut at a paragraph (else sentence) boundary near max_chars; return
    (clipped text, number of characters omitted)."""
    if len(text) <= max_chars:
        return text, 0
    window = text[:max_chars]
    cut = max_chars
    for sep, keep in (("\n\n", 0), (". ", 1)):
        i = window.rfind(sep)
        if i >= max_chars // 2:
            cut = i + keep
            break
    return text[:cut].rstrip(), len(text) - cut


def _papers_by_bibcode(con: duckdb.DuckDBPyConnection, bibcodes: list[str]) -> dict[str, tuple]:
    if not bibcodes:
        return {}
    rows = con.execute(
        "SELECT bibcode, arxiv_id, title, first_author, len(author), year, citation_count "
        f"FROM papers WHERE bibcode IN ({_in_list(len(bibcodes))})",
        bibcodes,
    ).fetchall()
    return {r[0]: r for r in rows}


def _paper_header(bibcode: str, papers: dict[str, tuple]) -> str:
    p = papers.get(bibcode)
    if not p:
        return f"## {bibcode} (not in papers table)"
    _, _, title, first_author, n_authors, year, cites = p
    return f'## {_author_label(first_author, n_authors)} {year or "n.d."} | "{title}" | {cites or 0} cit'


def _re2_escape(s: str) -> str:
    """Escape RE2 metacharacters (DuckDB's regex engine); re.escape also escapes
    characters RE2 may reject."""
    return re.sub(r"([\\.^$|?*+()\[\]{}])", r"\\\1", s)


# --------------------------------------------------------------------------- search


def _embed_queries(queries: list[str], model: str) -> list[list[float]]:
    from .embed import embed_texts, format_query, load_embedder  # lazy: pulls in mlx/transformers

    net, tokenizer = load_embedder(model, offline_first=True)
    return embed_texts(net, tokenizer, [format_query(q, model) for q in queries]).vectors.tolist()


def _top_chunks(
    con: duckdb.DuckDBPyConnection, model_id: str, qvec: list[float], n: int, since: int | None
) -> list[tuple[str, int, float]]:
    """Top-n (bibcode, chunk_index, cosine) by a full scan of chunk_embeddings.
    Filters sit inside the scan so they narrow the candidates *before* the top-n."""
    sql = (
        "SELECT e.bibcode, e.chunk_index, list_inner_product(e.embedding, ?::FLOAT[]) AS score "
        "FROM chunk_embeddings e WHERE e.model = ?"
    )
    params: list = [qvec, model_id]
    if since is not None:
        sql += " AND e.bibcode IN (SELECT bibcode FROM papers WHERE year >= ?)"
        params.append(since)
    sql += " ORDER BY score DESC, e.bibcode, e.chunk_index LIMIT ?"
    params.append(n)
    return con.execute(sql, params).fetchall()


def _compile_terms(terms: list[str], case: bool) -> tuple[str, re.Pattern | None]:
    """Combine --term regexes (OR). Returns (pattern for DuckDB/RE2, compiled Python
    regex for snippets -- None if Python rejects what RE2 accepted)."""
    pattern = "|".join(f"(?:{t})" for t in terms)
    try:
        rx = re.compile(pattern, 0 if case else re.IGNORECASE)
    except re.error:
        rx = None
    return pattern, rx


def _term_scan(
    con: duckdb.DuckDBPyConnection, model_id: str, qvecs: list[list[float]], pattern: str,
    case: bool, since: int | None, n: int,
) -> TermScan:
    """One regex scan of chunk text (+ heading path) picks the matching chunks; their
    cosine to every query variant is computed in the same pass, so the cost is one
    scan however many variants there are. Boilerplate chunks (acknowledgements,
    author lists, ...) are excluded: they are not embedded and are noise for a
    'which papers say X' question (use `grep` to include them)."""
    since_sql = " AND c.bibcode IN (SELECT bibcode FROM papers WHERE year >= ?)" if since is not None else ""
    scores = ", ".join("list_inner_product(e.embedding, ?::FLOAT[])" for _ in qvecs)
    sql = (
        "WITH m AS (SELECT c.bibcode, c.chunk_index FROM chunks c WHERE c.kind <> 'boilerplate' "
        f"AND regexp_matches(coalesce(c.heading_path, '') || ' ' || c.text, ?, ?){since_sql}) "
        f"SELECT m.bibcode, m.chunk_index, {scores} FROM m LEFT JOIN chunk_embeddings e "
        "ON e.bibcode = m.bibcode AND e.chunk_index = m.chunk_index AND e.model = ?"
    )
    params: list = [pattern, "c" if case else "i"] + ([since] if since is not None else []) + list(qvecs) + [model_id]
    try:
        rows = con.execute(sql, params).fetchall()
    except duckdb.Error as e:
        _die(
            f"--term regex rejected: {str(e).splitlines()[0]}\n"
            "(DuckDB uses RE2 syntax: no lookahead/lookbehind or backreferences. Use `|` for variants, "
            "`\\W*` between words, `\\b` for word boundaries.)"
        )
    ranked = []
    for qi in range(len(qvecs)):
        scored = [(r[0], r[1], r[2 + qi]) for r in rows if r[2 + qi] is not None]
        scored.sort(key=lambda t: (-t[2], t[0], t[1]))
        ranked.append(scored[:n])
    return TermScan(ranked, Counter(r[0] for r in rows), len(rows))


def _fuse(
    ranked: list[list[tuple[str, int, float]]], k: int, per_paper: int,
    term_ranked: list[list[tuple[str, int, float]]] | None = None, term_weight: float | None = None,
) -> list[Hit]:
    """Reciprocal-rank fusion across the query variants -- and, with --term, across the
    term-restricted rankings of the same variants -- then the per-paper cap. A passage
    that ranks well both ways is found twice and rises."""
    tw = TERM_WEIGHT if term_weight is None else term_weight
    rrf: dict[tuple[str, int], float] = {}
    best: dict[tuple[str, int], float] = {}
    seen: dict[tuple[str, int], list[int]] = {}
    termed: set[tuple[str, int]] = set()
    for qi, rows in enumerate(ranked, 1):
        for rank, (bibcode, idx, score) in enumerate(rows, 1):
            key = (bibcode, idx)
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank)
            best[key] = max(best.get(key, -1.0), score)
            seen.setdefault(key, []).append(qi)
    for rows in term_ranked or []:
        for rank, (bibcode, idx, score) in enumerate(rows, 1):
            key = (bibcode, idx)
            rrf[key] = rrf.get(key, 0.0) + tw / (RRF_K + rank)
            best[key] = max(best.get(key, -1.0), score)
            termed.add(key)
    hits: list[Hit] = []
    per: dict[str, int] = {}
    for key in sorted(rrf, key=lambda c: (-rrf[c], -best[c], c)):
        if per.get(key[0], 0) >= per_paper:
            continue
        per[key[0]] = per.get(key[0], 0) + 1
        hits.append(Hit(key[0], key[1], rrf[key], best[key], tuple(seen.get(key, ())), key in termed))
        if len(hits) >= k:
            break
    return hits


def _highlight(s: str, rx: re.Pattern | None) -> str:
    return rx.sub(lambda m: f"**{m.group(0)}**" if m.group(0) else "", s) if rx else s


def _preview(text: str, rx: re.Pattern | None, max_chars: int) -> str:
    """The passage head, clipped -- unless the term matches beyond the clip, in which
    case a shorter head plus keyword-in-context windows around the first matches, so
    the relevant sentence of a long passage is not cut off. Matches are **bold**."""
    clipped, omitted = _clip(text, max_chars)
    m = rx.search(text) if rx else None
    if m is None or m.end() <= len(clipped):
        return _highlight(clipped, rx) + (f" [+{omitted} chars]" if omitted else "")
    head, _ = _clip(text, max(200, max_chars // 4))
    half = max(120, (max_chars - len(head)) // 4)
    windows: list[str] = []
    shown, last_end = len(head), 0
    for mm in rx.finditer(text):
        if mm.end() == mm.start() or mm.start() < last_end:
            continue
        lo, hi = max(len(head), mm.start() - half), min(len(text), mm.end() + half)
        windows.append(("…" if lo > len(head) else "") + text[lo:hi].strip() + ("…" if hi < len(text) else ""))
        shown += hi - lo
        last_end = hi
        if len(windows) == 2:
            break
    if not windows:
        return _highlight(clipped, rx) + (f" [+{omitted} chars]" if omitted else "")
    body = "\n[…]\n".join([head] + [_highlight(w, rx) for w in windows])
    return body + f" [+{max(len(text) - shown, 0)} chars]"


def cmd_search(args: argparse.Namespace) -> int:
    ph: _Phases = args.phases
    if not 1 <= len(args.queries) <= MAX_QUERIES:
        _die(f"give 1-{MAX_QUERIES} queries (variants of the same question), got {len(args.queries)}")
    terms = [t for t in (args.term or []) if t]
    n_list = max(0, min(args.papers, MAX_LIST_PAPERS))
    con = _connect(args.db)
    model_id = args.model
    models = _models(con)
    if model_id not in models:
        have = ", ".join(f"{m} ({n} chunks)" for m, n in models.items()) or "none"
        _die(f"no embeddings for {model_id} in {args.db}; models present: {have}. Run `pixi run embed`.")
    eligible = con.execute("SELECT count(*) FROM chunks WHERE kind <> 'boilerplate'").fetchone()[0]
    if models[model_id] < COVERAGE_WARN * eligible:
        print(
            f"warning: only {models[model_id]}/{eligible} chunks are embedded "
            f"({100 * models[model_id] / eligible:.1f}%); newer papers may be missing",
            file=sys.stderr,
        )
    ph.mark("open")

    qvecs = _embed_queries(args.queries, model_id)
    ph.mark("embed queries (incl. import + model load)")

    n_candidates = max(100, 10 * args.k)
    ranked = [_top_chunks(con, model_id, qv, n_candidates, args.since) for qv in qvecs]
    ph.mark(f"score x{len(qvecs)}")

    scan: TermScan | None = None
    pattern = rx = None
    if terms:
        pattern, rx = _compile_terms(terms, args.case)
        scan = _term_scan(con, model_id, qvecs, pattern, args.case, args.since, n_candidates)
        ph.mark("term scan")

    hits = _fuse(ranked, args.k, args.per_paper, scan.ranked if scan else None)
    bibcodes = list(dict.fromkeys(h.bibcode for h in hits))
    extra: list[tuple[str, int]] = []
    if scan and n_list:
        extra = sorted(((b, n) for b, n in scan.per_paper.items() if b not in bibcodes), key=lambda t: (-t[1], t[0]))
    listed = extra[:n_list]
    papers = _papers_by_bibcode(con, bibcodes + [b for b, _ in listed])
    wanted = {(h.bibcode, h.idx) for h in hits}
    rows = con.execute(
        "SELECT bibcode, chunk_index, heading_path, section_heading, kind, text "
        f"FROM chunks WHERE bibcode IN ({_in_list(len(bibcodes))})",
        bibcodes,
    ).fetchall() if bibcodes else []
    chunks = {(r[0], r[1]): r for r in rows if (r[0], r[1]) in wanted}
    ph.mark("fetch hits")

    quoted = " | ".join(f'"{q}"' for q in args.queries)
    term_label = f"   term: {' OR '.join(repr(t) for t in terms)} ({'case-sensitive' if args.case else 'case-insensitive'})" if terms else ""
    print(f"# search: {quoted}{term_label}   ({len(hits)} passages, {len(bibcodes)} papers; score = best cosine across variants)")
    if scan:
        print(f"# term matched {scan.n_chunks} chunks in {len(scan.per_paper)} papers (non-boilerplate text + headings)")
        if scan.n_chunks == 0:
            print(
                "warning: --term matched nothing, so these results are semantic-only. Loosen the regex: allow "
                "spelling variants with `|`, put `\\W*` between words, drop \\b; test it with "
                "`grep --regex 'PATTERN'`."
            )
        elif scan.n_chunks > TERM_TOO_GENERIC:
            print(f"warning: --term matches {scan.n_chunks} chunks; that is too generic to sharpen the ranking. Use a more specific regex.")
    for bibcode in bibcodes:
        print(f"\n{_paper_header(bibcode, papers)}")
        arxiv = (papers.get(bibcode) or (None, None))[1]
        for h in (x for x in hits if x.bibcode == bibcode):
            row = chunks.get((h.bibcode, h.idx))
            if row is None:
                continue
            _, _, heading_path, section_heading, kind, text = row
            tag = f" arXiv:{arxiv}" if arxiv else ""
            via = ",".join(map(str, h.queries))
            via = f"q{via}" if via else "term-only"
            via += " +term" if h.term and h.queries else ""
            print(f"- `{h.bibcode}#{h.idx}`{tag} | {h.score:.2f} | {via} | {kind}")
            print(f"  {heading_path or section_heading or '(no heading)'}")
            print(_preview(text, rx if h.term else None, args.max_chars))
    if scan and listed:
        more = len(extra) - len(listed)
        print(f"\n## Other papers matching the term ({len(extra)}; listing {len(listed)} by number of matching chunks)")
        for b, n in listed:
            p = papers.get(b)
            if p:
                print(f'- `{b}` | {_author_label(p[3], p[4])} {p[5] or "n.d."} | "{p[2]}" | {n} chunk{"s" if n != 1 else ""}')
            else:
                print(f"- `{b}` | {n} chunks")
        if more > 0:
            print(f"({more} more not listed: raise --papers (max {MAX_LIST_PAPERS}), narrow the term, or use `grep`)")
    if hits:
        print(f"\n---\nExpand: chunk '{hits[0].bibcode}#{hits[0].idx}' --context 1 | outline: paper '{hits[0].bibcode}'")
    return 0


# --------------------------------------------------------------------------- chunk / paper


def _parse_id(s: str) -> tuple[str, int]:
    bibcode, sep, idx = s.rpartition("#")
    if not sep or not bibcode or not idx.isdigit():
        _die(f"bad chunk id {s!r}; expected BIBCODE#INDEX (quote it: bibcodes contain '&')")
    return bibcode, int(idx)


def cmd_chunk(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    ids = [_parse_id(s) for s in args.ids[:8]]
    papers = _papers_by_bibcode(con, list(dict.fromkeys(b for b, _ in ids)))
    for bibcode, idx in ids:
        rows = con.execute(
            "SELECT chunk_index, heading_path, section_heading, kind, n_tokens, text FROM chunks "
            "WHERE bibcode = ? AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
            [bibcode, idx - args.context, idx + args.context],
        ).fetchall()
        if not any(r[0] == idx for r in rows):
            _die(f"no chunk {bibcode}#{idx}")
        print(_paper_header(bibcode, papers))
        for i, heading_path, section_heading, kind, n_tokens, text in rows:
            clipped, omitted = _clip(text, args.max_chars)
            mark = "►" if i == idx else " "
            print(f"\n{mark} `{bibcode}#{i}` | {heading_path or section_heading or '(no heading)'} | {kind} | {n_tokens} tok")
            print(clipped + (f" [+{omitted} chars]" if omitted else ""))
        print()
    if args.context:
        print("(neighbouring chunks can be tables: a section's tables follow its prose)")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    row = con.execute(
        "SELECT bibcode, arxiv_id, title, author, year, pub, citation_count, abstract FROM papers "
        "WHERE bibcode = ? OR list_contains(alternate_bibcode, ?) LIMIT 1",
        [args.bibcode, args.bibcode],
    ).fetchone()
    if row is None:
        _die(f"no paper {args.bibcode!r} in the papers table")
    bibcode, arxiv_id, title, authors, year, pub, cites, abstract = row
    authors = authors or []
    shown = ", ".join(authors[:6]) + (f", +{len(authors) - 6} more" if len(authors) > 6 else "")
    print(f"# {title}\n{shown}\n{year or 'n.d.'} | {pub or 'preprint'} | {cites or 0} citations | `{bibcode}`"
          + (f" | arXiv:{arxiv_id}" if arxiv_id else ""))

    if args.chunks or args.full:
        lo, hi = 0, 10**9
        if args.chunks:
            a, _, b = args.chunks.partition("-")
            if not a.isdigit() or (b and not b.isdigit()):
                _die("--chunks expects A or A-B (chunk indices)")
            lo, hi = int(a), int(b or a)
        rows = con.execute(
            "SELECT chunk_index, heading_path, section_heading, kind, text FROM chunks "
            "WHERE bibcode = ? AND chunk_index BETWEEN ? AND ? AND kind <> 'boilerplate' ORDER BY chunk_index",
            [bibcode, lo, hi],
        ).fetchall()
        if not rows:
            _die("no chunks in that range (boilerplate sections are skipped)")
        used = 0
        print()
        for i, heading_path, section_heading, kind, text in rows:
            if used + len(text) > args.max_chars and used:
                print(f"\n[stopped before chunk {i}: --max-chars reached; continue with --chunks {i}-{rows[-1][0]}]")
                break
            print(f"[#{i}] {heading_path or section_heading or ''} | {kind}\n{text}\n")
            used += len(text)
        return 0

    print(f"\n## Abstract\n{abstract or '(none)'}")
    outline = con.execute(
        "SELECT coalesce(heading_path, section_heading, '(front matter)') AS h, min(chunk_index), max(chunk_index), "
        "sum(n_tokens), bool_or(kind = 'table'), bool_and(kind = 'boilerplate') "
        "FROM chunks WHERE bibcode = ? GROUP BY 1 ORDER BY min(chunk_index)",
        [bibcode],
    ).fetchall()
    if not outline:
        print("\n(no full text indexed for this paper)")
        return 0
    print("\n## Outline (heading | chunks | tokens)")
    for h, lo, hi, tokens, has_table, only_boiler in outline:
        flags = (" [tables]" if has_table else "") + (" [boilerplate]" if only_boiler else "")
        print(f"- {h} | #{lo}-{hi} | {tokens} tok{flags}")
    print(f"\nRead a section: paper '{bibcode}' --chunks A-B   (whole text: --full, capped by --max-chars)")
    return 0


# --------------------------------------------------------------------------- grep / status


def _snippet(text: str, rx: re.Pattern | None, width: int = 110) -> str:
    start = end = 0
    m = rx.search(text) if rx else None
    if m:
        start, end = m.span()
    lo, hi = max(0, start - width), min(len(text), end + width)
    return ("…" if lo else "") + " ".join(text[lo:hi].split()) + ("…" if hi < len(text) else "")


def cmd_grep(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    flag = "c" if args.case else "i"
    if args.regex or args.word:
        pat = args.pattern if args.regex else _re2_escape(args.pattern)
        if args.word:
            pat = rf"\b(?:{pat})\b"
        cond, params = "regexp_matches(text, ?, ?)", [pat, flag]
    else:
        pat = re.escape(args.pattern)
        if args.case:
            cond, params = "contains(text, ?)", [args.pattern]
        else:
            cond, params = "contains(lower(text), ?)", [args.pattern.lower()]
    try:
        rx = re.compile(pat, 0 if args.case else re.IGNORECASE)
    except re.error:
        rx = None
    sql = f"SELECT bibcode, chunk_index, kind, text FROM chunks WHERE {cond}"
    if args.paper:
        sql += " AND bibcode = ?"
        params.append(args.paper)
    if args.since is not None:
        sql += " AND bibcode IN (SELECT bibcode FROM papers WHERE year >= ?)"
        params.append(args.since)
    if args.also:
        # paper-level co-occurrence: the paper must mention this somewhere (any chunk)
        sql += " AND bibcode IN (SELECT bibcode FROM chunks WHERE regexp_matches(text, ?, 'i'))"
        params.append(args.also)
    sql += " ORDER BY bibcode, chunk_index LIMIT ?"
    params.append(GREP_ROW_CAP)
    try:
        rows = con.execute(sql, params).fetchall()
    except duckdb.Error as e:
        _die(f"pattern rejected: {str(e).splitlines()[0]} (DuckDB uses RE2 syntax: no lookarounds/backreferences)")

    by_paper: dict[str, list[tuple]] = {}
    for r in rows:
        by_paper.setdefault(r[0], []).append(r)
    ordered = sorted(by_paper, key=lambda b: (-len(by_paper[b]), b))
    papers = _papers_by_bibcode(con, ordered[: max(args.papers, 1)])
    capped = " (capped: more matches exist)" if len(rows) >= GREP_ROW_CAP else ""
    opts = "".join(
        [" (regex)" if args.regex else "", " (whole word)" if args.word else "", " (case-sensitive)" if args.case else "",
         f" since {args.since}" if args.since is not None else "", f" also-mentioning {args.also!r}" if args.also else ""]
    )
    print(f"# grep: {args.pattern!r}{opts}: {len(rows)} chunks in {len(by_paper)} papers{capped}")
    if not rows:
        return 0
    print("\n## Papers by number of matching chunks")
    for b in ordered[: args.papers]:
        p = papers.get(b)
        print(f"- `{b}` | {len(by_paper[b])} chunks | {(p[2] if p else '?')} ({p[5] if p else '?'})")
    if len(ordered) > args.papers:
        print(f"({len(ordered) - args.papers} more papers not listed; raise --papers)")
    print("\n## Matches")
    shown = 0
    for b in ordered:
        for _, idx, kind, text in by_paper[b][: args.per_paper]:
            if shown >= args.limit:
                break
            print(f"- `{b}#{idx}` | {kind}: {_snippet(text, rx)}")
            shown += 1
    if len(rows) > shown:
        print(f"\n({len(rows) - shown} more matching chunks not shown; raise --limit/--per-paper or narrow with --paper)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    q = lambda sql: con.execute(sql).fetchall()
    print(f"database: {args.db} ({args.db.stat().st_size / 1e9:.2f} GB)")
    print(f"papers: {q('SELECT count(*) FROM papers')[0][0]}")
    chunk_rows = q("SELECT kind, count(*), count(DISTINCT bibcode), sum(n_tokens) FROM chunks GROUP BY 1 ORDER BY 1")
    print(f"chunked papers: {q('SELECT count(DISTINCT bibcode) FROM chunks')[0][0]}")
    for kind, n, _, tokens in chunk_rows:
        print(f"  chunks[{kind}]: {n} ({tokens} tokens)")
    eligible = con.execute("SELECT count(*) FROM chunks WHERE kind <> 'boilerplate'").fetchone()[0]
    models = _models(con)
    if not models:
        print("embeddings: none (run `pixi run embed`)")
        return 0
    for model, n in models.items():
        dim = con.execute("SELECT len(embedding) FROM chunk_embeddings WHERE model = ? LIMIT 1", [model]).fetchone()[0]
        print(f"embeddings[{model}]: {n} vectors, {dim}-d, coverage {100 * n / eligible:.1f}% of {eligible} non-boilerplate chunks")
    orphans = con.execute(
        "SELECT count(*) FROM chunk_embeddings e WHERE NOT EXISTS "
        "(SELECT 1 FROM chunks c WHERE c.bibcode = e.bibcode AND c.chunk_index = e.chunk_index)"
    ).fetchone()[0]
    print(f"orphan embeddings (chunk gone; run `pixi run embed` to clean up): {orphans}")
    stale = con.execute(
        "SELECT count(*) FROM chunk_embeddings e JOIN chunks c "
        "ON c.bibcode = e.bibcode AND c.chunk_index = e.chunk_index WHERE c.kind = 'boilerplate'"
    ).fetchone()[0]
    if stale:
        print(f"stale embeddings of chunks now tagged boilerplate (run `pixi run embed` to drop them): {stale}")
    return 0


# --------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=config.DB_PATH, help="database file (default: papers.duckdb)")
    common.add_argument("--timing", action="store_true", help="print per-phase timings to stderr")

    p = argparse.ArgumentParser(prog="arxivrag-search", description="Search the local astro-ph literature index")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", parents=[common], help="semantic search (1-4 query variants), optionally fused with a regex --term")
    s.add_argument("queries", nargs="+", help="variants of one question; each is embedded and results are fused")
    s.add_argument(
        "--term", action="append", metavar="REGEX",
        help="regex (case-insensitive; repeatable = OR) that passages should contain. Use for any named line, "
        "object, number or acronym, with spelling variants, e.g. 'ca\\W*(ii)?\\W*triplet'. Fuses a term-restricted "
        "ranking with the semantic one, shows matches in context, and lists other matching papers.",
    )
    s.add_argument("--case", action="store_true", help="make --term case-sensitive (e.g. 'CaT' without matching 'cat')")
    s.add_argument("--k", type=int, default=8, help="max passages returned")
    s.add_argument("--per-paper", type=int, default=2, help="max passages per paper")
    s.add_argument("--max-chars", type=int, default=1200, help="preview length per passage")
    s.add_argument(
        "--papers", type=int, default=DEFAULT_LIST_PAPERS,
        help=f"with --term: how many other matching papers to list after the passages (default {DEFAULT_LIST_PAPERS}, "
        f"max {MAX_LIST_PAPERS}, 0 = none). Raise it when you expect many.",
    )
    s.add_argument("--since", type=int, default=None, help="only papers from this year on")
    s.add_argument("--model", default=config.EMBED_MODEL, choices=list(config.EMBED_MODELS), help="embedding model (default config.EMBED_MODEL)")
    s.set_defaults(func=cmd_search)

    c = sub.add_parser("chunk", parents=[common], help="show passage(s) with neighbours")
    c.add_argument("ids", nargs="+", help="BIBCODE#INDEX (max 8)")
    c.add_argument("--context", type=int, default=0, help="neighbouring chunks on each side")
    c.add_argument("--max-chars", type=int, default=6000, help="per-chunk character cap")
    c.set_defaults(func=cmd_chunk)

    d = sub.add_parser("paper", parents=[common], help="paper metadata, abstract and outline; or its text")
    d.add_argument("bibcode")
    d.add_argument("--chunks", default=None, help="read chunk A or range A-B")
    d.add_argument("--full", action="store_true", help="read the whole text (capped by --max-chars)")
    d.add_argument("--max-chars", type=int, default=20000)
    d.set_defaults(func=cmd_paper)

    g = sub.add_parser("grep", parents=[common], help="exhaustive literal/regex text match over chunk text")
    g.add_argument("pattern")
    g.add_argument("--regex", action="store_true", help="treat PATTERN as a regular expression")
    g.add_argument("--word", action="store_true", help="whole-word match (wraps PATTERN in \\b...\\b)")
    g.add_argument("--case", action="store_true", help="case-sensitive (default: insensitive)")
    g.add_argument("--since", type=int, default=None, help="only papers from this year on")
    g.add_argument("--also", metavar="REGEX", default=None, help="keep only papers that also mention this regex somewhere (e.g. 'little red dot')")
    g.add_argument("--paper", default=None, help="restrict to one bibcode")
    g.add_argument("--papers", type=int, default=50, help="papers listed in the summary")
    g.add_argument("--limit", type=int, default=30, help="max snippets shown")
    g.add_argument("--per-paper", type=int, default=3, help="max snippets per paper")
    g.set_defaults(func=cmd_grep)

    t = sub.add_parser("status", parents=[common], help="what is indexed and embedded")
    t.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    args = _parser().parse_args(argv)
    args.phases = _Phases(args.timing)
    rc = args.func(args)
    args.phases.report()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

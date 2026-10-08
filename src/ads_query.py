"""
Build a paper + citation-edge test corpus from a single seed paper.

Fetches the seed paper's own metadata (bibcode is the row key -- not every
paper in a citation graph has an arXiv ID, so bibcode has to be the anchor;
arxiv_id is stored alongside as the key we'll actually use for Arcana
uploads once a paper has one). ADS returns each record's `reference` and
`citation` fields as bibcode lists directly, so no extra references()/
citations() search calls are needed: one query for the seed, one or two
batched bibcode:(... OR ...) queries for its direct neighbors.

Edges are stored as directed (citing_bibcode, cited_bibcode) pairs, star
topology only: the seed's direct references and direct citations. Cross-
links among the neighbors themselves are not fetched (would require a
reference/citation pull per neighbor -- a depth-2 expansion, not done here).

Run directly:
    pixi run query --bibcode "2025A&A...702A..57H"
    pixi run query --query 'abs:"Little Red Dot"'   # many seeds at once
    pixi run query-lrd                              # shortcut for the LRD search above (first step of `pixi run all`)
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import duckdb
import httpx

import config

_ARXIV_ID_RE = re.compile(r'arXiv:(\d{4}\.\d{4,5}|\d{7})', re.IGNORECASE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    bibcode           VARCHAR PRIMARY KEY,
    alternate_bibcode VARCHAR[],
    arxiv_id          VARCHAR,
    title             VARCHAR,
    author            VARCHAR[],
    first_author      VARCHAR,
    abstract          VARCHAR,
    year              INTEGER,
    pubdate           VARCHAR,
    doctype           VARCHAR,
    pub               VARCHAR,
    volume            VARCHAR,
    page              VARCHAR[],
    doi               VARCHAR[],
    keyword           VARCHAR[],
    arxiv_class       VARCHAR[],
    citation_count    INTEGER,
    property          VARCHAR[],
    query_source      VARCHAR,
    added_at          VARCHAR
);

CREATE TABLE IF NOT EXISTS citation_edges (
    citing_bibcode VARCHAR,
    cited_bibcode  VARCHAR,
    PRIMARY KEY (citing_bibcode, cited_bibcode)
);
"""

_PAPER_COLUMNS = [
    'bibcode',
    'alternate_bibcode',
    'arxiv_id',
    'title',
    'author',
    'first_author',
    'abstract',
    'year',
    'pubdate',
    'doctype',
    'pub',
    'volume',
    'page',
    'doi',
    'keyword',
    'arxiv_class',
    'citation_count',
    'property',
    'query_source',
    'added_at',
]

# Named column list, not bare VALUES(...): fetch.py ALTER TABLEs in its own
# `esources` column, so a positional insert against "all columns" breaks
# the moment the table grows.
_UPSERT_PAPERS = f"""
INSERT INTO papers ({', '.join(_PAPER_COLUMNS)})
VALUES ({', '.join(['?'] * len(_PAPER_COLUMNS))})
ON CONFLICT (bibcode) DO UPDATE SET
    alternate_bibcode = excluded.alternate_bibcode,
    arxiv_id          = excluded.arxiv_id,
    title             = excluded.title,
    author            = excluded.author,
    first_author      = excluded.first_author,
    abstract          = excluded.abstract,
    year              = excluded.year,
    pubdate           = excluded.pubdate,
    doctype           = excluded.doctype,
    pub               = excluded.pub,
    volume            = excluded.volume,
    page              = excluded.page,
    doi               = excluded.doi,
    keyword           = excluded.keyword,
    arxiv_class       = excluded.arxiv_class,
    citation_count    = excluded.citation_count,
    property          = excluded.property
"""

_INSERT_EDGE = """
INSERT INTO citation_edges VALUES (?, ?)
ON CONFLICT (citing_bibcode, cited_bibcode) DO NOTHING
"""


def _session() -> httpx.Client:
    if not config.ADS_API_KEY:
        raise SystemExit('ADS_API_KEY is not set (put it in .env or the environment)')
    return httpx.Client(
        headers={'Authorization': f'Bearer {config.ADS_API_KEY}'}, timeout=30
    )


def _search(client: httpx.Client, q: str, rows: int) -> list[dict]:
    resp = client.get(
        f'{config.ADS_BASE_URL}/search/query',
        params={'q': q, 'fl': config.ADS_FIELDS, 'rows': rows},
    )
    resp.raise_for_status()
    return resp.json()['response']['docs']


def _fetch_by_bibcode(client: httpx.Client, bibcode: str) -> dict | None:
    docs = _search(client, f'bibcode:"{bibcode}"', rows=1)
    return docs[0] if docs else None


def _fetch_many_by_bibcode(client: httpx.Client, bibcodes: list[str]) -> list[dict]:
    """Batch-fetch full metadata for a list of bibcodes, chunked to keep query length sane."""
    docs: list[dict] = []
    for i in range(0, len(bibcodes), config.ADS_BATCH_SIZE):
        chunk = bibcodes[i : i + config.ADS_BATCH_SIZE]
        q = 'bibcode:(' + ' OR '.join(f'"{b}"' for b in chunk) + ')'
        docs.extend(_search(client, q, rows=len(chunk)))
        time.sleep(config.ADS_DELAY)
    return docs


def _extract_arxiv_id(identifiers: list[str] | None) -> str | None:
    if not identifiers:
        return None
    for ident in identifiers:
        m = _ARXIV_ID_RE.search(ident)
        if m:
            return m.group(1)
    return None


def _doc_to_params(d: dict, query_source: str) -> tuple:
    year = d.get('year')
    return (
        d.get('bibcode'),
        d.get('alternate_bibcode') or [],
        _extract_arxiv_id(d.get('identifier')),
        (d.get('title') or [None])[0],
        d.get('author') or [],
        d.get('first_author'),
        d.get('abstract'),
        int(year) if year else None,
        d.get('pubdate'),
        d.get('doctype'),
        d.get('pub'),
        d.get('volume'),
        d.get('page') or [],
        d.get('doi') or [],
        d.get('keyword') or [],
        d.get('arxiv_class') or [],
        d.get('citation_count'),
        d.get('property') or [],
        query_source,
        datetime.now(timezone.utc).isoformat(),
    )


def build_test_corpus(seed_bibcode: str) -> None:
    con = duckdb.connect(str(config.DB_PATH))
    con.execute(_SCHEMA)

    with _session() as client:
        print(f'Fetching seed paper: {seed_bibcode}')
        seed = _fetch_by_bibcode(client, seed_bibcode)
        if seed is None:
            print(f'  [error] no ADS record found for bibcode {seed_bibcode!r}')
            return

        references = seed.get('reference') or []
        citations = seed.get('citation') or []
        print(f'  → {len(references)} references, {len(citations)} citations')

        print(f'Fetching metadata for {len(references)} referenced papers...')
        ref_docs = _fetch_many_by_bibcode(client, references)
        print(f'  → got {len(ref_docs)} records')

        print(f'Fetching metadata for {len(citations)} citing papers...')
        cite_docs = _fetch_many_by_bibcode(client, citations)
        print(f'  → got {len(cite_docs)} records')

    rows = [_doc_to_params(seed, 'seed')]
    rows += [_doc_to_params(d, f'reference_of:{seed_bibcode}') for d in ref_docs]
    rows += [_doc_to_params(d, f'citation_of:{seed_bibcode}') for d in cite_docs]

    con.executemany(_UPSERT_PAPERS, rows)

    edges = [(seed_bibcode, r) for r in references]  # seed cites r
    edges += [(c, seed_bibcode) for c in citations]  # c cites seed
    con.executemany(_INSERT_EDGE, edges)
    con.commit()

    n_papers = con.execute('SELECT count(*) FROM papers').fetchone()[0]
    n_edges = con.execute('SELECT count(*) FROM citation_edges').fetchone()[0]
    print(
        f'\nDone. papers={n_papers} (total in DB), citation_edges={n_edges} → {config.DB_PATH}'
    )
    con.close()


def build_corpus_from_search(
    query: str,
    max_seeds: int = 2000,
    pull_references: bool = True,
    pull_citations: bool = True,
) -> None:
    """
    Same star-topology trick as build_test_corpus, but with many seed papers
    at once (an ADS query) instead of one bibcode. Each seed's own
    reference/citation bibcode lists come back in the same search response
    (no per-seed follow-up call needed).

    Only neighbor bibcodes not already present in papers.duckdb get
    batch-fetched -- this makes it safe (and cheap) to re-run the exact same
    query repeatedly, e.g. weekly with no date restriction at all: already-
    known seeds just get their metadata refreshed (picking up newly-
    accumulated citations), already-known neighbors are skipped entirely,
    and only genuinely new bibcodes cost an ADS call.

    pull_references/pull_citations let a weekly "citations only" run skip
    references entirely, since references mostly reconverge on literature
    already ingested rather than surfacing anything new.
    """
    con = duckdb.connect(str(config.DB_PATH))
    con.execute(_SCHEMA)

    known_bibcodes = {
        r[0] for r in con.execute('SELECT bibcode FROM papers').fetchall()
    }

    with _session() as client:
        print(f'Searching: {query!r}')
        seeds = _search(client, query, rows=max_seeds)
        print(f'  → {len(seeds)} seed papers')

        seed_bibcodes = {d['bibcode'] for d in seeds if d.get('bibcode')}

        neighbor_bibcodes: set[str] = set()
        edges: list[tuple[str, str]] = []
        for d in seeds:
            bib = d.get('bibcode')
            if not bib:
                continue
            refs = (d.get('reference') or []) if pull_references else []
            cites = (d.get('citation') or []) if pull_citations else []
            neighbor_bibcodes.update(refs)
            neighbor_bibcodes.update(cites)
            edges += [(bib, r) for r in refs]  # seed cites r
            edges += [(c, bib) for c in cites]  # c cites seed

        already_have = known_bibcodes | seed_bibcodes
        to_fetch = sorted(neighbor_bibcodes - already_have)
        print(
            f'  → {len(neighbor_bibcodes)} distinct referenced/citing bibcodes '
            f'({len(to_fetch)} new, {len(neighbor_bibcodes) - len(to_fetch)} already in the database)'
        )

        print(f'Fetching metadata for {len(to_fetch)} neighbor papers...')
        neighbor_docs = _fetch_many_by_bibcode(client, to_fetch)
        print(f'  → got {len(neighbor_docs)} records')

    rows = [_doc_to_params(d, f'search:{query}') for d in seeds]
    rows += [_doc_to_params(d, f'expansion_of_search:{query}') for d in neighbor_docs]

    con.executemany(_UPSERT_PAPERS, rows)
    con.executemany(_INSERT_EDGE, edges)
    con.commit()

    n_papers = con.execute('SELECT count(*) FROM papers').fetchone()[0]
    n_edges = con.execute('SELECT count(*) FROM citation_edges').fetchone()[0]
    print(
        f'\nDone. papers={n_papers} (total in DB), citation_edges={n_edges} → {config.DB_PATH}'
    )
    con.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build a citation-graph corpus from ADS'
    )
    parser.add_argument(
        '--bibcode', help='Single seed bibcode, e.g. "2025A&A...702A..57H"'
    )
    parser.add_argument(
        '--query', help='ADS query for many seeds, e.g. \'abs:"Little Red Dot"\''
    )
    parser.add_argument(
        '--no-references',
        action='store_true',
        help='With --query, skip pulling references (citations only)',
    )
    parser.add_argument(
        '--no-citations',
        action='store_true',
        help='With --query, skip pulling citations (references only)',
    )
    parser.add_argument(
        '--no-canonicalize',
        action='store_true',
        help='Skip the final pass that merges papers ADS has re-keyed (see canonicalize.py)',
    )
    args = parser.parse_args()

    if args.bibcode:
        build_test_corpus(seed_bibcode=args.bibcode)
    elif args.query:
        build_corpus_from_search(
            query=args.query,
            pull_references=not args.no_references,
            pull_citations=not args.no_citations,
        )
    else:
        parser.error('pass --bibcode or --query')

    if not args.no_canonicalize:
        # ADS re-keys records over time (preprint -> journal bibcode, gained arXiv ids), which
        # leaves twin rows and stale ids behind; fold every paper back onto its canonical bibcode.
        from .canonicalize import canonicalize

        print('\nCanonicalizing against ADS...')
        canonicalize(apply=True)

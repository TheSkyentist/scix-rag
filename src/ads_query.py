"""
Build a paper + citation-edge corpus from ADS: one or many seed papers plus
their direct references and citations.

The seeds come from an ADS search (--query) or a single bibcode (--bibcode,
which is just the query `bibcode:"..."`). Bibcode is the row key -- not every
paper in a citation graph has an arXiv ID, so bibcode has to be the anchor;
arxiv_id is stored alongside as the key `fetch` uses for on-disk filenames
once a paper has one. ADS returns each record's `reference` and `citation`
fields as bibcode lists directly, so no extra references()/citations() search
calls are needed: the seed search carries the neighbor lists, and one batched
bibcode:(... OR ...) query per 75 neighbors fetches their metadata.

Edges are stored as directed (citing_bibcode, cited_bibcode) pairs, star
topology only: each seed's direct references and direct citations. Cross-
links among the neighbors themselves are not fetched (would require a
reference/citation pull per neighbor -- a depth-2 expansion, not done here).

Run directly:
    pixi run query --bibcode "2025A&A...702A..57H"
    pixi run query --query 'abs:"Little Red Dot"'   # many seeds at once
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


def _get_response(client: httpx.Client, params: dict) -> dict:
    resp = client.get(f'{config.ADS_BASE_URL}/search/query', params=params)
    if resp.status_code == 429:
        raise SystemExit(
            'ADS rate limit reached (ADS allows about 5000 requests per day per API key); '
            're-run later'
        )
    resp.raise_for_status()
    return resp.json()['response']


def _search(client: httpx.Client, q: str, rows: int) -> list[dict]:
    return _get_response(client, {'q': q, 'fl': config.ADS_FIELDS, 'rows': rows})[
        'docs'
    ]


def _search_seeds(client: httpx.Client, q: str, max_seeds: int) -> list[dict]:
    """Page through a search, up to max_seeds records; warn if the search has more."""
    docs: list[dict] = []
    found = 0
    while len(docs) < max_seeds:
        response = _get_response(
            client,
            {
                'q': q,
                'fl': config.ADS_FIELDS,
                'rows': min(config.ADS_PAGE_SIZE, max_seeds - len(docs)),
                'start': len(docs),
            },
        )
        found = response['numFound']
        docs.extend(response['docs'])
        if not response['docs'] or len(docs) >= found:
            break
        time.sleep(config.ADS_DELAY)
    if found > len(docs):
        print(
            f'  [warning] the search matches {found} papers but only the first {len(docs)} '
            f'were used as seeds; raise --max-seeds or narrow the query'
        )
    return docs


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


def _read_bibcode_file(path: str) -> list[str]:
    """One bibcode per line; blank lines and `#` comments (whole-line or trailing) are ignored."""
    lines = (line.split('#', 1)[0].strip() for line in open(path, encoding='utf-8'))
    return [line for line in lines if line]


def _known_bibcodes(dry_run: bool) -> set[str]:
    """Bibcodes already in papers.duckdb; a dry run never creates or writes the file."""
    if dry_run and not config.DB_PATH.exists():
        return set()
    con = duckdb.connect(str(config.DB_PATH), read_only=dry_run)
    try:
        return {r[0] for r in con.execute('SELECT bibcode FROM papers').fetchall()}
    except duckdb.CatalogException:  # fresh database, no papers table yet
        return set()
    finally:
        con.close()


def _batches(n: int, size: int) -> int:
    return -(-n // size)


def build_corpus(
    query: str | None = None,
    bibcodes: list[str] | None = None,
    max_seeds: int = 2000,
    pull_references: bool = False,
    pull_citations: bool = False,
    dry_run: bool = False,
    canonicalize_after: bool = True,
) -> None:
    """
    Add the seeds -- every paper matching an ADS query plus any listed bibcodes
    (the union) -- to papers.duckdb, and with pull_references/pull_citations the
    papers each seed cites / is cited by. Each seed's own reference/citation
    bibcode lists come back in the same record (no per-seed follow-up call);
    only the neighbors' metadata costs extra requests. At most max_seeds seeds
    are taken from the query.

    Only neighbor bibcodes not already present in papers.duckdb get
    batch-fetched -- this makes it safe (and cheap) to re-run: already-known
    seeds just get their metadata refreshed (picking up newly-accumulated
    citations), already-known neighbors are skipped entirely, and only
    genuinely new bibcodes cost an ADS call.

    dry_run does the searches but writes nothing, and prints what a real run
    would add.
    """
    bibcodes = list(dict.fromkeys(bibcodes or []))
    known_bibcodes = _known_bibcodes(dry_run)

    seeds: dict[str, tuple[dict, str]] = {}  # bibcode -> (ADS record, query_source)
    with _session() as client:
        if query:
            print(f'Searching: {query!r}')
            for d in _search_seeds(client, query, max_seeds):
                if d.get('bibcode'):
                    seeds.setdefault(d['bibcode'], (d, f'search:{query}'))
            print(f'  → {len(seeds)} seed papers from the search')
        if bibcodes:
            print(f'Fetching {len(bibcodes)} listed bibcodes...')
            found = _fetch_many_by_bibcode(client, bibcodes)
            answered = {
                b
                for d in found
                for b in [d.get('bibcode'), *(d.get('alternate_bibcode') or [])]
            }
            missing = [b for b in bibcodes if b not in answered]
            for d in found:
                if d.get('bibcode'):
                    seeds.setdefault(d['bibcode'], (d, 'seed'))
            print(f'  → got {len(found)} records')
            if missing:
                shown = ', '.join(missing[:20]) + (' ...' if len(missing) > 20 else '')
                print(
                    f'  [warning] ADS has no record for {len(missing)} bibcodes, skipped: {shown}'
                )
        if not seeds:
            raise SystemExit('No ADS records found for the seeds; nothing to add')

        # bibcode -> query_source of the first seed that introduced it
        neighbors: dict[str, str] = {}
        edges: list[tuple[str, str]] = []
        for bib, (d, _) in seeds.items():
            refs = (d.get('reference') or []) if pull_references else []
            cites = (d.get('citation') or []) if pull_citations else []
            for r in refs:
                neighbors.setdefault(r, f'reference_of:{bib}')
            for c in cites:
                neighbors.setdefault(c, f'citation_of:{bib}')
            edges += [(bib, r) for r in refs]  # seed cites r
            edges += [(c, bib) for c in cites]  # c cites seed

        to_fetch = sorted(set(neighbors) - known_bibcodes - set(seeds))
        n_seed_requests = _batches(len(bibcodes), config.ADS_BATCH_SIZE) + (
            _batches(len(seeds), config.ADS_PAGE_SIZE) if query else 0
        )
        if pull_references or pull_citations:
            print(
                f'  → {len(neighbors)} distinct referenced/citing bibcodes '
                f'({len(to_fetch)} new, {len(neighbors) - len(to_fetch)} already in the database or seeds)'
            )

        if dry_run:
            total = len(known_bibcodes | set(seeds) | set(to_fetch))
            n_requests = (
                n_seed_requests
                + _batches(len(to_fetch), config.ADS_BATCH_SIZE)
                + (_batches(total, config.ADS_BATCH_SIZE) if canonicalize_after else 0)
            )
            print('\nDry run, nothing written. A real run would add:')
            print(f'  {len(seeds):,} seeds ({len(set(seeds) - known_bibcodes):,} new)')
            if pull_references or pull_citations:
                print(
                    f'  {len(neighbors):,} neighbors ({len(to_fetch):,} new, '
                    f'{len(neighbors) - len(to_fetch):,} already known)'
                )
            print(f'  about {n_requests:,} ADS requests')
            return

        neighbor_docs: list[dict] = []
        if to_fetch:
            print(f'Fetching metadata for {len(to_fetch)} neighbor papers...')
            neighbor_docs = _fetch_many_by_bibcode(client, to_fetch)
            print(f'  → got {len(neighbor_docs)} records')

    rows = [_doc_to_params(d, source) for d, source in seeds.values()]
    # ADS may hand back a record under a different (canonical) bibcode than the one listed
    rows += [
        _doc_to_params(d, neighbors.get(d.get('bibcode'), 'expansion'))
        for d in neighbor_docs
    ]

    con = duckdb.connect(str(config.DB_PATH))
    con.execute(_SCHEMA)
    con.executemany(_UPSERT_PAPERS, rows)
    if edges:  # executemany rejects an empty parameter list
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
        description='Build a citation-graph corpus from ADS: seeds (a search and/or bibcodes), '
        'optionally plus their references and citations'
    )
    parser.add_argument(
        '--query',
        help='ADS search whose matches are seeds, e.g. \'abs:"Little Red Dots"\'',
    )
    parser.add_argument(
        '--bibcode',
        action='append',
        default=[],
        help='Seed bibcode, e.g. "2025A&A...702A..57H" (repeatable)',
    )
    parser.add_argument(
        '--bibcode-file',
        help='Text file of seed bibcodes, one per line (# comments allowed)',
    )
    parser.add_argument(
        '--max-seeds',
        type=int,
        default=2000,
        help='Most seed papers to take from --query (default 2000); a warning is printed if the search has more',
    )
    parser.add_argument(
        '--references', action='store_true', help='Also add the papers each seed cites'
    )
    parser.add_argument(
        '--citations',
        action='store_true',
        help='Also add the papers that cite each seed',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run the searches and print what would be added, without changing the database',
    )
    parser.add_argument(
        '--no-canonicalize',
        action='store_true',
        help='Skip the final pass that merges papers ADS has re-keyed (see canonicalize.py)',
    )
    args = parser.parse_args()

    bibcodes = list(args.bibcode)
    if args.bibcode_file:
        try:
            bibcodes += _read_bibcode_file(args.bibcode_file)
        except OSError as e:
            parser.error(f'cannot read --bibcode-file: {e}')
    if not (args.query or bibcodes):
        parser.error('pass --query, --bibcode or --bibcode-file')

    build_corpus(
        query=args.query,
        bibcodes=bibcodes,
        max_seeds=args.max_seeds,
        pull_references=args.references,
        pull_citations=args.citations,
        dry_run=args.dry_run,
        canonicalize_after=not args.no_canonicalize,
    )

    if not args.no_canonicalize and not args.dry_run:
        # ADS re-keys records over time (preprint -> journal bibcode, gained arXiv ids), which
        # leaves twin rows and stale ids behind; fold every paper back onto its canonical bibcode.
        from .canonicalize import canonicalize

        print('\nCanonicalizing against ADS...')
        canonicalize(apply=True)

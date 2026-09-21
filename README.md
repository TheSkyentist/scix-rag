# arXivRAG

A retrieval-augmented generation system over a curated subset of astro-ph
papers, built on GWDG's Arcana/RAG service. This document is the working
plan: what we know about the provider, the decisions we've made and why,
what's still open, and where the repo actually stands.

## The provider: GWDG Arcana

Docs: <https://docs.hpc.gwdg.de/services/ai-services/arcana/index.html>

Arcana is a managed RAG service: you upload files, it runs them through
**Docling** to convert to Markdown, indexes them, and an LLM answers
questions against the index with citations.

**Ingestion**
- Accepted upload formats: `.txt`, `.md`, `.docx`/`.dotx`/`.docm`/`.dotm`,
  `.pptx`/`.potx`/`.ppsx`/`.pptm`/`.potm`/`.ppsm`, `.pdf`, `.html`/`.htm`/`.xhtml`.
  Everything is converted to Markdown internally — you are never uploading
  "raw" content, Docling always gets a pass at it.
- 300 MB hard limit per file, 50 MB recommended for reasonable indexing time.
- **Indexing is synchronous per upload call** — the POST blocks until that
  file's index is built. This is the real bottleneck for bulk ingestion,
  not file size or storage.
- No documented cap on total files or total storage per Arcana. Unresolved —
  see [Open questions](#open-questions).

**Docling's internal chunking** (only relevant if we ever upload PDFs
directly instead of pre-cleaned Markdown): it annotates the converted
Markdown with `[Page (n)]`, `[Y: n]` (vertical position, page scaled to
1000 lines, 5 zones/page), and `[Split]` markers, and those `[Split]`
markers drive chunk boundaries. Each uploaded file also gets a Docling-
generated YAML header: `Author, Title, Description, Filename, Extension,
Number of Pages, Version`. This is file-level, not per-chunk.

**Querying**: a response comes back with citations — source **filename**
and **page number**, nothing richer. Retrieval looks like a fixed top-5
passages (only temperature/top_p are documented as tunable, not top-k).
No documented way to filter retrieval by metadata.

**Models**: embedding model is `qwen3-embedding-4B` (confirmed directly by
GWDG, not documented anywhere public — 32K context, Matryoshka embeddings
from 32–2560 dims). Recommended chat model for RAG use is
`Qwen 3 30B A3B Instruct 2507` (tool-calling capable; plain
`Meta Llama 3.1 8B Instruct` and external models are explicitly not
recommended/supported for Arcana use).

## Decision: arXiv HTML5, cleaned ourselves, one file per paper

We are **not** trusting Docling on raw PDF or raw arXiv HTML. Instead:
source is arXiv's HTML5 service (`arxiv.org/html/{arxiv_id}`, ar5iv as
fallback — arXiv's own LaTeXML conversion is ~97% "produces something,"
~75% error-free as of recent reporting), and we run our own HTML → clean
Markdown pass before it ever reaches Arcana.

Why preprocess instead of just uploading the raw HTML, measured on
`2506.05459v1` (a real, math-heavy RUBIES paper):

| | bytes | % of file |
|---|---|---|
| Raw arXiv HTML | 530,662 | 100% |
| — of which MathML markup | 131,494 | 25% |
| Plain readable text | ~119,000 | 22% |
| Head/nav/script/style boilerplate | ~3,500 | <1% |

File size itself was never going to be a problem — even the raw HTML is
three orders of magnitude under the 300 MB cap. The reason to clean it
ourselves is **what gets embedded**, not size:
- Every `<math>` tag carries an `alttext` attribute holding the original
  LaTeX (confirmed: `alttext="\sim"`). Swapping MathML trees for `$alttext$`
  cuts that 25% of the file down to almost nothing and gives the embedding
  model LaTeX instead of an XML soup.
- Nav/head/citation-link scaffolding (419 `ltx_ref` anchors, 138 `<cite>`
  tags) renders down to plain `(Author Year)` text rather than being
  indexed as markup.
- We keep real section headings from `<section>`/`<hN>` under our own
  control, rather than trusting Docling's generic re-chunking of an
  already-idiosyncratic LaTeXML export.

Checked and rejected: scraping the HTML bibliography for the citation
graph. Reference-list entries are plain bibliographic strings
(`Author, Year, Journal, Vol, Page`); only some carry a bare `arXiv:...`
string, none carry bibcodes or DOIs consistently. The citation graph comes
from the ADS/SciX API instead (see below), which resolves this properly.

Upload unit is **one Markdown file per paper**, not per chunk — see
[Metadata](#metadata-architecture) for why.

## Metadata architecture

Arcana's own metadata surface is thin and file-scoped: a Docling-generated
YAML header per file, and at query time just `source filename + page`.
It has no concept of authors, citations, or cross-paper structure — that
has to live in our own store, with the filename as the only bridge back.

**Bridge**: at upload time, filename = the paper's stable key. We use
`arxiv_id` when present, since it's assigned once and never changes,
unlike a bibcode (which can be replaced when a preprint becomes a
published record). When Arcana cites "source file X, page N," we join `X`
straight back into our own DB to recover everything else.

**Caveat found empirically**: not every paper in a citation graph has an
arXiv ID — e.g. `1996A&AS..117..393B` (SExtractor) predates arXiv entirely.
7 of 187 papers in our first pilot corpus have no arXiv ID. So `arxiv_id`
can't be the primary key of the *graph*, only the upload key for papers we
actually ingest full text for. ADS's own `bibcode` is universal (every
record has exactly one), so it's the row's real primary key; `bibcode`
churn on the arXiv-preprint-to-published transition is handled via ADS's
`alternate_bibcode` field, stored alongside for reconciliation.

### Schema (`data/papers.duckdb`)

```sql
CREATE TABLE papers (
    bibcode           VARCHAR PRIMARY KEY,   -- ADS canonical id, universal
    alternate_bibcode VARCHAR[],             -- e.g. preprint bibcode if this is the published one
    arxiv_id          VARCHAR,               -- nullable; the Arcana upload/join key when present
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
    property          VARCHAR[],             -- e.g. REFEREED, OPENACCESS
    query_source      VARCHAR,               -- how it entered the corpus; first-write-wins on upsert
    added_at          VARCHAR
);

CREATE TABLE citation_edges (
    citing_bibcode VARCHAR,
    cited_bibcode  VARCHAR,
    PRIMARY KEY (citing_bibcode, cited_bibcode)
);
```

Chosen over plain Parquet because the citation graph is inherently
relational (many-to-many edges) and DuckDB gives native `LIST` columns
plus real SQL joins over a single file, while still exporting to Parquet
trivially if needed. Upserts on `bibcode` refresh metadata (e.g.
`citation_count` changes over time) but never touch `query_source` or
`added_at` once set.

Built by `src/ads_query.py` (`pixi run query --bibcode "..."`). ADS/SciX
confirms the same API/base URL/auth as classic ADS
(`api.adsabs.harvard.edu`), 5,000 requests/day (not a constraint at this
scale), and — usefully — `reference`/`citation` bibcode lists are
queryable **fields** on a record (`fl=reference,citation`), so one query
for the seed plus a couple of batched `bibcode:(... OR ...)` calls covers
a full depth-1 neighborhood without per-paper reference/citation search
calls.

**Design choice**: edges are star topology only — the seed's direct
references and direct citations, not the cross-links among the neighbors
themselves. Capturing those would mean pulling `reference`/`citation` for
every neighbor too (cheap, since it's the same batched query), deferred
until we actually need a denser subgraph.

## Pilot corpus

Seed: `2025A&A...702A..57H` (Hviding et al., RUBIES little-red-dots census,
arXiv:2506.05459). Run via `pixi run query --bibcode "2025A&A...702A..57H"`:

- 90 references + 100 citations (1 citation bibcode had no ADS record) + seed
- **187 papers, 190 citation edges** in `data/papers.duckdb`
- 180/187 have a resolvable arXiv ID (the 7 without are pre-arXiv or
  instrumentation papers — exactly the case bibcode-as-PK was for)
- No null titles/abstracts across the set

This is also the working test set for the HTML-cleaning and Arcana-upload
steps below.

## Repo layout

| Path | Status |
|---|---|
| `config.py` | Current. ADS lean field set, `DB_PATH` for DuckDB. Arcana env vars optional until we ingest. |
| `src/ads_query.py` | Current. Builds the citation-graph test corpus described above. |
| `src/fetch_html.py` | **Stale** — reads/writes `papers.parquet`, predates the DuckDB schema. Needs rework to read seed set from `papers.duckdb`. |
| `src/chunk.py` | **Stale** — assumes per-chunk hierarchical splitting + per-chunk upload, which we're moving away from (one file per paper now). Needs a rewrite as an HTML→clean-Markdown converter instead. |
| `src/arcana_ingest.py` | **Stale** — uploads one file per chunk with YAML frontmatter baked into the body text. Superseded by the one-file-per-paper approach; needs rework once an Arcana collection exists. |
| `notebooks/explore.py` | Marimo scratch notebook. |

## Open questions

Things that need a live Arcana pilot or a direct question to GWDG (same as
we did for the embedding model) rather than more reading:

- Total storage / file-count limits per Arcana — undocumented.
- Real per-file indexing latency — needed to size the eventual corpus
  against the synchronous-upload bottleneck.
- Whether a YAML frontmatter block we add to our own uploaded Markdown is
  respected as Docling's metadata header, or overwritten by Docling's own
  extraction.
- Whether top-k retrieval (fixed at 5 in the documented UI settings) is
  configurable via the API.
- Env var naming for Arcana: `config.py` currently accepts either
  `ARCANA_TOKEN` or `SAIA_API_KEY`, plus `ARCANA_NAME` (the Arcana/
  collection ID) — **`ARCANA_NAME` doesn't exist yet and needs to be
  created and added to `.env` before any ingestion step.**
- Final corpus scope (which subfield, how many papers) — deferred until
  indexing throughput is measured, rather than guessed.

## Next steps

1. Write the HTML → clean-Markdown converter and run it across the 180
   arXiv-ID papers in the pilot corpus (validates math/citation handling
   at more than one-paper scale).
2. Create an actual Arcana collection, add `ARCANA_NAME` + token to
   `.env`, and test upload/indexing latency on a handful of cleaned files.
3. Use that latency measurement to decide final corpus scope.
4. Rework `fetch_html.py` / `chunk.py` / `arcana_ingest.py` to read from
   `papers.duckdb` and upload one file per paper instead of per chunk.

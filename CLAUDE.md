# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A local, self-hosted retrieval-augmented-generation pipeline over a citation-graph-selected subset of astro-ph papers (currently seeded from "Little Red Dots" literature via ADS/SciX). The corpus is built by walking the citation graph, then each paper's full text is fetched and cleaned into Markdown. Embedding/retrieval is the active next step (see "Planned next stage" below) — everything through Markdown conversion is built and running in production (a weekly GitHub Action keeps the corpus growing).

This project previously targeted GWDG's hosted Arcana RAG service for embedding/indexing; that integration has been dropped in favor of a fully local pipeline (own embedding model, own vector search) after Arcana's indexing step destructively wiped a previously-working index on a large batch timeout. Don't resurrect `arcana_upload.py`/`ARCANA_*` config or re-introduce an upload-to-Arcana step.

## Commands

Environment/task management is [pixi](https://pixi.sh); each pipeline stage lives in its own pixi feature/environment (separate dependency sets — e.g. the PDF path pulls in `torch` via MinerU, which the HTML-only stages don't need).

```
pixi run query --bibcode "2025A&A...702A..57H"   # seed a citation-graph corpus from one paper
pixi run query --query 'abs:"Little Red Dots"'   # seed from an ADS search instead (many seeds at once)
pixi run fetch [--limit N]                       # fetch raw HTML/PDF for papers not yet converted
pixi run convert [--limit N]                     # convert data/html + data/pdf into data/markdown
```

`pixi run query`'s `--no-references`/`--no-citations` flags restrict which side of the citation graph gets pulled (the weekly Action runs citations-only, since references mostly reconverge on literature already ingested). All three stages are safe to re-run repeatedly — see "Resumability" below.

There is no test suite, linter, or build step configured in this repo currently.

## Architecture

### Pipeline stages (in order)

1. **`src/ads_query.py`** — queries the ADS/SciX API, builds/updates the citation graph in `data/papers.duckdb`. Two entry points: `build_test_corpus` (one seed bibcode, pulls its direct references + citations — star topology, no cross-links between neighbors) and `build_corpus_from_search` (many seeds from an ADS search query at once, same star-topology logic per seed). Re-running is cheap and safe: only bibcodes not already in the DB get fetched.
2. **`src/fetch.py`** — for every paper in `papers.duckdb` that has no Markdown/HTML/PDF on disk yet, fetches raw content: arXiv HTML5 (then ar5iv fallback) for papers with an `arxiv_id`, otherwise ADS's own hosted PDF sources (tried in `PUB_PDF → ADS_PDF → EPRINT_PDF` priority order, via ADS's link-gateway). HTML is validated by actually running it through the real converter before being kept (falls back to PDF if conversion produces too little text) — but the Markdown from that validation pass is thrown away; `convert.py` owns the copy that's actually kept.
3. **`src/convert.py`** — converts whatever's sitting in `data/html/`/`data/pdf/` into Markdown in `data/markdown/`, deleting the raw file on success. Purely filesystem-driven (no DB reads/writes). HTML path uses `src/html_to_markdown.py` (pure Python, no ML — see below). PDF path shells out to MinerU (`mineru-kit`, standard tier, `--ocr-mode auto` per-page since even nominally born-digital PDFs can have image-only pages) and then strips the references section, base64-embedded figure crops, and ADS watermark lines. Files that convert to fewer than `config.MIN_MARKDOWN_CHARS` are left in place for manual review rather than retried automatically.
4. **`src/html_to_markdown.py`** — the actual HTML→Markdown cleaning logic `convert.py`'s HTML path calls. Recursively walks LaTeXML's `<section>`/`ltx_para` structure into heading + paragraph blocks, swaps every `<math alttext="...">` for inline/display LaTeX (avoids embedding raw MathML), renders figures as caption-only (images are never embedded, just their captions) and tables as pipe-delimited text, and drops the bibliography entirely (the citation graph lives in `citation_edges`, not embedded prose).

### Resumability ("done" tracking)

Deliberately simple, filesystem-first design: a paper is considered fetched/converted once `data/markdown/{stem}.md` (or `data/html/{stem}.html` or `data/pdf/{stem}.pdf`) exists, full stop. `fetch.py`/`convert.py` do not write pipeline-status columns to `papers.duckdb` — the one exception is `esources` (which full-text sources ADS claims to have per paper), which is real ADS metadata worth caching, not pipeline bookkeeping. This trades a small amount of wasted retry effort (an unfetchable paper gets retried every run, no permanent "gave up" marker) for every stage being trivially safe to interrupt and re-run.

`stem` is `arxiv_id` when present, else `bibcode` — this is also the filename convention used everywhere under `data/`.

### Data model (`data/papers.duckdb`)

Two tables:
- **`papers`** — one row per paper, keyed on `bibcode` (ADS's universal ID; not every paper has an `arxiv_id`, e.g. pre-arXiv instrumentation papers, so bibcode has to be the real primary key). `arxiv_id` is nullable and is the actual join key used for on-disk filenames. `alternate_bibcode` absorbs bibcode churn from preprint→published transitions. `query_source` records how a paper entered the corpus and is first-write-wins on upsert (never overwritten on re-query). Upserts refresh everything else (e.g. `citation_count` changes over time).
- **`citation_edges`** — directed `(citing_bibcode, cited_bibcode)` pairs, star topology only (a seed's direct references/citations, not cross-links between neighbors).

`data/papers.duckdb` is the one artifact meant to be committed to git (see `.gitignore`) — `data/html/`, `data/markdown/`, `data/pdf/` are ephemeral local working files, regenerated by re-running `fetch`/`convert`, and are gitignored.

## Planned next stage: chunking + embedding (design decided, not yet built)

The next stage splits each paper's Markdown into passage-sized chunks and embeds them for retrieval — replacing what Arcana used to do internally. Locked-in design, ahead of implementation:

- **Embedding model**: Qwen3-Embedding-4B (32K context), run locally (pixi's `embed` feature already has `mlx`/`mlx-lm` as dependencies for this — no task/script wired up yet).
- **Chunk boundaries**: walk the Markdown's own heading structure, pack consecutive paragraphs within a section up to `config.MAX_CHUNK_TOKENS` (1024), cut only at paragraph boundaries, no overlap between adjacent chunks. Token counts use Qwen3's own tokenizer, not a proxy.
- **Embedded-text prefix**: each chunk gets the paper title + its section heading prepended before embedding (disambiguates otherwise-generic-sounding passages); the prefix is reconstructed at embed time from other columns, not duplicated into stored chunk text.
- **Storage**: a new `chunks` table in `papers.duckdb` (`bibcode`, `arxiv_id`, `chunk_index`, `section_heading`, `text`, `n_tokens`, `embedding`), using DuckDB's VSS extension for the vector column — no separate vector store.
- Papers with no Markdown on disk yet (still pending fetch/convert, or in MinerU's PDF queue) are simply skipped at embed time, same skip-if-not-ready pattern the rest of the pipeline already uses.

## Known-stale docs

`README.md` still describes the original GWDG-Arcana-based plan (one-file-per-paper upload, `arcana_ingest.py`, old module names like `fetch_html.py`) and predates the current pipeline — don't treat it as authoritative for current architecture.

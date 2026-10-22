# scix-rag

A locally hosted retrieval-augmented-generation (RAG) system for querying astronomy papers from [NASA ADS / SciX](https://scixplorer.org). You choose a topic, and it builds a corpus by walking the citation graph from your seed papers. It then downloads and cleans their full text, and indexes it so you can search it by meaning.

It is built to be used by agentic AI systems first. An agent (i.e. Claude Code, opencode, etc.) calls a read-only search tool, gets back passages with their ADS bibcodes, and writes the answer itself. The same search is available as a command-line tool for manual use. Everything runs on your own machine: your own embedding model, your own index, no hosted RAG service.

## How it works

The pipeline has six stages. Each one resumes where it left off, so you can re-run it whenever you want to grow the corpus.

| Stage     | What it does                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| `query`   | Seeds a corpus from an ADS search or a single paper, then pulls in the references and citations of each seed. |
| `fetch`   | Downloads the full text of each paper, as HTML where possible and as PDF otherwise.                           |
| `convert` | Cleans the HTML or PDF into Markdown.                                                                         |
| `chunk`   | Splits each paper into passage-sized chunks along its section structure.                                      |
| `embed`   | Turns each chunk into a vector with a local embedding model.                                                  |
| `search`  | Finds the chunks most relevant to a query, and prints them with their paper IDs.                              |

## Requirements

- **An Apple Silicon Mac.** Embedding runs on [MLX](https://github.com/ml-explore/mlx), which only runs on Apple Silicon, and the pixi environment is currently locked to `osx-arm64`. MLX is Apple's array framework for machine learning on its own chips. It runs the embedding model fast on the Mac's GPU, so no separate GPU or server is needed.
- **[pixi](https://pixi.sh).** It manages every dependency. Each stage has its own environment, so the heavy PDF-conversion dependencies stay out of the lighter stages.
- **[DuckDB](https://duckdb.org).** It stores the papers, the citation graph, the chunks and the vectors in one local file, and computes vector similarity inside the database. That means there is no separate vector-database server to run.
- **An ADS API key.** It is free; you can create one in your [ADS/SciX](https://scixplorer.org/user/settings/token) account settings. Copy `.env.example` to `.env` and put the key in it, as described under [Stage 1](#stage-1-query).

## Disclaimers

1. **The text is often from the preprint, not the published paper.** Full text is fetched as HTML from the arXiv preprint when one exists. It can differ from the final published version, so check the journal version before you quote or cite a number.
2. **Not every paper can be fetched.** Papers with no arXiv preprint need a publisher or ADS-hosted PDF, and many don't have one. Some of those are behind a paywall or block automated downloads. Not every PDF converts cleanly either. Papers that can't be fetched or converted are left out of the index, so a search finding nothing is not evidence that no paper says it.
3. **This is not a replacement for reading the papers.** Retrieval tells you where to look. It does not make you an expert in the literature, and it can miss relevant passages. Read the papers and do your own work.

## Stage 1: query

`query` is the only stage that talks to the ADS API, and the only one that needs your key. Put it in a `.env` file at the repo root (gitignored):

```
cp .env.example .env
# then edit .env so it reads: ADS_API_KEY=your-token-here
```

A run does two things: it picks **seeds**, and it optionally pulls in the **references and citations** of those seeds. Everything is stored in `data/papers.duckdb` (`data/` is gitignored). A paper that is in the database is picked up by every later stage, so it gets fetched, converted, chunked and embedded.

### Choose the seeds

Seeds come from an [ADS search](https://ui.adsabs.harvard.edu/help/search/search-syntax), from bibcodes, or from both at once (the seeds are the union):

```
# Every paper matching an ADS search. Any ADS search syntax works.
pixi run query --query 'abs:"Little Red Dots"'
pixi run query --query 'abs:"Little Red Dots" year:2024-'
pixi run query --query 'docs(library/YOUR_LIBRARY_ID)'      # a library you curated on ADS

# Specific papers
pixi run query --bibcode "2025A&A...702A..57H"
pixi run query --bibcode "2025A&A...702A..57H" --bibcode "2026Sci...391..767M"
pixi run query --bibcode-file seeds.txt                      # one bibcode per line, `#` comments allowed
```

Quote bibcodes, because `&` means something to the shell. A bibcode ADS has no record of is reported by name and skipped, and the run goes on with the rest.

### Pull in references and citations

By default only the seeds are added. To also add the papers around them, say which side of the citation graph you want:

```
pixi run query --bibcode "2025A&A...702A..57H" --references              # papers the seed cites
pixi run query --bibcode "2025A&A...702A..57H" --citations               # papers that cite the seed
pixi run query --query 'abs:"Little Red Dots"' --references --citations  # both
```

This works the same way for a search and for bibcodes. It goes one step out from each seed. It does not follow the links between the neighbors, so the corpus is the seeds plus their direct neighbors and no further.

### Preview before you commit

Pulling in references and citations can add far more papers than the seeds themselves, and every added paper costs download, conversion and embedding time later. `--dry-run` runs the searches and prints what a real run would add, without changing the database:

```
pixi run query --query 'abs:"Little Red Dots"' --references --citations --dry-run
# ...
# Dry run, nothing written. A real run would add:
#   1,412 seeds (1,412 new)
#   9,380 neighbors (7,915 new, 1,465 already known)
#   about 220 ADS requests
```

The request count includes the [clean-up pass](#keeping-one-row-per-paper) at the end. The numbers above are illustrative.

### What gets stored

`query` creates the database `data/papers.duckdb` on its first run, with two tables. Later stages add two more.

- **`papers`**: one row per paper, keyed on its bibcode. It holds the title, authors, abstract, year, journal, DOI, keywords, citation count, and the arXiv ID when ADS knows one. A `query_source` column records how the paper entered the corpus: `seed`, `search:<query>`, `reference_of:<bibcode>` or `citation_of:<bibcode>`. The first entry is kept when a later query finds the same paper. `fetch` adds an `esources` column listing which full-text sources ADS has for the paper.
- **`citation_edges`**: one row per citation, as a (citing bibcode, cited bibcode) pair. Only the links between a seed and its neighbors are stored.

Like the rest of `data/`, the database is gitignored, so back it up yourself.

### Other options

| Option | Effect |
| --- | --- |
| `--max-seeds N` | Most seeds a `--query` may use (default 2000). A warning says how many matches were left out. |
| `--no-canonicalize` | Skip the clean-up pass at the end that merges papers ADS has re-keyed (see [below](#keeping-one-row-per-paper)). |

### Keeping one row per paper

ADS changes its records over time. Usually a paper first appears as an arXiv preprint (`2026arXiv260104955C`). When it is published, ADS re-keys it to the journal record (`2026Natur.657..621C`) and the old bibcode turns into an alias. Less often, a journal record gains an arXiv ID after the fact. Left alone, the database would end up with two rows for one paper, or with a stale arXiv ID. The `canonicalize` step fixes this. It looks every paper up on ADS, groups the rows that are the same paper, and keeps one row under ADS's canonical bibcode. When it merges rows it also:

- repoints the citation links to the surviving row
- keeps one copy of the paper's chunks and embeddings, without re-embedding
- moves or deletes raw files stored under the old name
- fixes the fetch and convert ignore lists, so a paper that gained an arXiv ID is retried from arXiv

Every `query` run ends with it unless you pass `--no-canonicalize`, and `fetch --refresh` runs it for papers that still have no arXiv ID. You can also run it on its own:

```
pixi run canonicalize                    # dry run: print what would change
pixi run canonicalize --apply            # write the changes
pixi run canonicalize --since 2015 --apply   # only look up papers from 2015 on
```

It costs one ADS lookup for every paper in the database, in batches of 75, so about 150 requests for 11,000 papers. That cost is part of every `query` run. It only merges rows that ADS itself links, through an alias or a shared arXiv ID. A preprint and a journal record that ADS has not linked stay as two rows, and there are a few dozen of these in the corpus this was developed on.

### Good to know

- Re-running is cheap and safe. Seeds already in the database are refreshed, only new neighbors cost an ADS request, and you can add as many different queries as you like to one corpus.
- ADS allows about 5000 requests per day per key. A large expansion uses a few hundred. If you hit the limit, the run stops with a message and you can re-run it the next day.

## Stage 2: fetch

`fetch` downloads the full text of every paper in the database that doesn't have it yet. It makes no difference whether a paper came in as a seed or as a reference or citation of one. It only downloads. Cleaning the text into Markdown is the next stage. Where the text comes from depends on the paper:

- **Papers with an arXiv ID:** the arXiv HTML version, then the ar5iv HTML version. A page only counts if it converts to a reasonable amount of text. If neither does, `fetch` falls back to the arXiv PDF.
- **Papers without an arXiv ID:** a PDF that ADS links to, tried in this order: the publisher's PDF, the PDF ADS hosts itself, the preprint PDF. These are fetched through ADS's link gateway.

The raw files land in `data/html/` and `data/pdf/`.

```
pixi run fetch --dry-run                     # how many papers are pending, and by which route, without downloading
pixi run fetch --limit 20                    # try the first 20 pending papers, as a smoke test
pixi run fetch                               # everything pending
pixi run fetch --retry-failed                # also retry papers a previous run gave up on
pixi run fetch --refresh --dry-run           # report what a refresh would change (below)
pixi run fetch --refresh                     # refresh the arXiv links first, then fetch
pixi run fetch --refresh --since 2015        # refresh only the papers published in 2015 or later
```

`--dry-run` also reports how many papers the ignore list would skip (see below). At the end of a real run `fetch` prints a summary: how many papers were fetched as HTML, how many as PDF, how many had nothing fetchable, and how many hit a temporary error and will be tried again next time.

### Papers that can't be fetched

When every source for a paper definitively answers "no", `fetch` writes the paper to `data/fetch_ignore.txt` by itself, one line per paper with the reason:

```
2019ApJ...000..000X  # no ADS PDF source (esources: none)
2301.00000  # no arXiv HTML/PDF
```

Later runs skip every paper on that list, so the same dead ends aren't retried every time. To try a paper again, delete its line, or run `pixi run fetch --retry-failed` to retry the whole list. Network errors, timeouts and rate limits are never written to the list, so those papers are retried on the next run. Like everything under `data/`, the list is gitignored, so back it up along with the database.

### Refreshing arXiv links

Most papers have an arXiv ID from the start. Some journal records, though, are not yet linked to their preprint when you run `query`, and ADS may add the link later. `--refresh` asks ADS again about the papers that have no arXiv ID, and updates those that gained one, so they are fetched from arXiv's HTML instead of a PDF. Papers ADS has re-keyed are merged into one row, the same [clean-up](#keeping-one-row-per-paper) `query` ends with. `--since YEAR` limits the refresh to recent papers, and `--dry-run` reports what would change without writing anything.

### Good to know

- Fetching is resumable. A paper counts as done once a file for it exists under `data/`, so you can stop a run with Ctrl-C and start it again.
- It waits at least a second between papers on purpose, so a large corpus takes hours. It is safe to leave running and stop whenever.
- The PDFs fetched from publishers are the publishers' copyrighted files. They are kept locally for your own research, so don't redistribute the contents of `data/`.

## Browsing the database

`browse` opens a [marimo](https://marimo.io) notebook in your browser, with the corpus attached read-only so nothing you do there can change it:

```
pixi run browse
```

- **Browse a table:** pick a table, the rows per page and a page number. Every table is available: `papers`, `citation_edges`, `chunks` and `chunk_embeddings`. Paging is fast even deep into the 345,000 chunks.
- **Run your own SQL:** the last cell is a SQL editor. Start from the example (the 100 most-cited papers) and change it.
- **Vectors:** the browser leaves out the `embedding` column of `chunk_embeddings`, because 1,024 numbers per row are unreadable. Your own SQL can still select it.
- **Write lock:** like `search`, it can't open the database while `embed` or `chunk` is running.
- **It's a plain Python file:** `notebooks/browse.py` is yours to edit or extend.

## License

GPL-3.0. See [LICENSE](LICENSE).

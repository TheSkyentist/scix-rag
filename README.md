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
- **`scix-search` on your `PATH`**, to search from any directory or from an agent. Link the script into a folder on your `PATH`, for example `ln -s "$PWD/bin/scix-search" ~/.local/bin/`, and check that the folder really is on your `PATH`. See [Stage 6](#stage-6-search).
- **The skill in your agent's skills folder**, to let an agent use the search. For Claude Code, `ln -s "$PWD/skills/scix-search" ~/.claude/skills/scix-search`. Other agents can read `skills/scix-search/SKILL.md` or have its contents pasted into their instructions file. See [Using it from an AI agent](#using-it-from-an-ai-agent).

## Disclaimers

1. **The text is often from the preprint, not the published paper.** Full text is fetched as HTML from the arXiv preprint when one exists. It can differ from the final published version, so check the journal version before you quote or cite a number.
2. **Not every paper can be fetched.** Papers with no arXiv preprint need a publisher or ADS-hosted PDF, and many don't have one. Some of those are behind a paywall or block automated downloads. Not every PDF converts cleanly either. Papers that can't be fetched or converted are left out of the index, so a search finding nothing is not evidence that no paper says it.
3. **This is not a replacement for reading the papers.** Retrieval tells you where to look. It does not make you an expert in the literature, and it can miss relevant passages. Read the papers and do your own work.

## Quick start

```
git clone https://github.com/TheSkyentist/scix-rag.git && cd scix-rag
cp .env.example .env                                        # then put your ADS API key in .env
pixi run query --query 'abs:"your topic"' --dry-run         # preview how big the corpus would be
pixi run query --query 'abs:"your topic"'                   # add the papers to the database
pixi run update                                             # fetch, convert, chunk and embed them
bin/scix-search search --semantic 'a question about your topic'
```

`pixi run update` runs the four stages after `query` in order, and stops at the first one that fails. Every stage resumes where it left off, so you can stop it and run it again. The first run of each stage downloads its environment, which is large for `convert` because of the PDF tools, and `embed` downloads the embedding model. To try the pipeline cheaply first, seed it from a single paper, `pixi run query --bibcode '2025A&A...702A..57H'`, and run `update` on that. The stages are described one by one below, together with their options, and [Requirements](#requirements) says how to put `scix-search` on your `PATH`.

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

### Adding a paper by hand

If a paper can't be fetched but you have its PDF, from your library, your institution or the publisher, put it in `data/pdf/` yourself. Name it after the paper's stem: its arXiv ID when it has one, otherwise its bibcode, for example `1953JChPh..21.1087M.pdf`. The lines in `fetch_ignore.txt` show the right name for each paper. `fetch` sees the file and skips the paper, and `convert` turns the PDF into Markdown like any downloaded one. The paper must already be in the database, so add it with `pixi run query --bibcode` first if it isn't. After that, run `convert` and the later stages as usual.

### Refreshing arXiv links

Most papers have an arXiv ID from the start. Some journal records, though, are not yet linked to their preprint when you run `query`, and ADS may add the link later. `--refresh` asks ADS again about the papers that have no arXiv ID, and updates those that gained one, so they are fetched from arXiv's HTML instead of a PDF. Papers ADS has re-keyed are merged into one row, the same [clean-up](#keeping-one-row-per-paper) `query` ends with. `--since YEAR` limits the refresh to recent papers, and `--dry-run` reports what would change without writing anything.

### Good to know

- Fetching is resumable. A paper counts as done once a file for it exists under `data/`, so you can stop a run with Ctrl-C and start it again.
- It waits at least a second between papers on purpose, so a large corpus takes hours. It is safe to leave running and stop whenever.
- The PDFs fetched from publishers are the publishers' copyrighted files. They are kept locally for your own research, so don't redistribute the contents of `data/`.

## Stage 3: convert

`convert` turns the raw files `fetch` downloaded into Markdown, one file per paper in `data/markdown/`, named by the paper's arXiv ID or, without one, its bibcode. Once a paper's Markdown is written the raw file is deleted. It only looks at the files in `data/html/` and `data/pdf/` and never touches the database. It runs in its own pixi environment, because the PDF tools are large.

- **HTML** (from arXiv and ar5iv) is cleaned by plain Python code, with no machine learning, so it is fast and reliable. The Markdown keeps the headings, paragraphs, lists and tables (as text), and writes math as LaTeX. Figures are kept as captions only, because images are never stored. The bibliography is dropped, since the citation links are in the database.
- **PDF** is converted by [MinerU](https://github.com/opendatalab/MinerU), which runs locally, with OCR on pages that are only images. It is the slowest part of the whole pipeline: an HTML paper takes moments, while a PDF can take up to its 30-minute limit, so a corpus with many PDF-only papers can spend a long time here. It is also less reliable: math, tables and scanned older papers can come out imperfect. Afterwards the reference list, embedded image data and ADS page stamps are removed. MinerU is a separate tool, so check its license terms for your use.

```
pixi run convert --dry-run                        # how many files are waiting, without converting
pixi run convert --limit 5                        # try the first 5 of each type, as a smoke test
pixi run convert                                  # everything waiting
pixi run convert 1965ApJS....9..321M.pdf          # convert specific files, even if they are on the ignore list
pixi run convert --retry-failed                   # also retry every file on the ignore list
```

At the end of a run it prints how many files were converted and how many failed.

### Papers that can't be converted

A file that converts to almost no text (under 500 characters), makes MinerU time out, or makes it exit with an error is left where it is, and `convert` adds its filename and the reason to `data/convert_ignore.txt` by itself:

```
1965ApJS....9..321M.pdf  # MinerU timeout (1800s)
```

Later runs skip the files on that list, so a PDF that always times out doesn't cost 30 minutes every time. Naming a file on the command line converts it anyway and never records anything, and `--retry-failed` retries the whole list. Other errors, such as a disk problem, are only printed and are tried again on the next run. The list is gitignored like the rest of `data/`, so back it up with the database. Unlike `fetch_ignore.txt`, it holds filenames with their extension.

## Stage 4: chunk

`chunk` splits each paper's Markdown into passage-sized pieces, and the later stages work on those pieces. Search returns chunks, so their size sets how precise an answer can be. It reads `data/markdown/` and the `papers` table, and writes the `chunks` table. It needs no GPU and is quick: a few minutes for about 11,000 papers. The first run downloads the embedding model's tokenizer, a small download with no model weights, because chunk sizes are counted in the same tokens the embedding model uses.

**How it splits.** It follows each paper's headings and packs the paragraphs of a section into chunks of up to 1,024 tokens. It cuts only between paragraphs, with no overlap between neighbouring chunks. A paragraph that is too long is split at line breaks, then at bullet points, then at sentences. Each chunk records its heading and the full path of headings above it, for example `2 Method > 2.3 Raman Scattering`, so a generic heading like "Simulation" keeps its parent's context. The embed stage puts the paper title and this path in front of the text.

Every chunk has a `kind`:

| Kind | What it is |
| --- | --- |
| `prose` | Ordinary text. |
| `table` | One whole table with its caption. It is stored in full, but only the caption and header row are embedded. |
| `boilerplate` | Acknowledgements, data availability, funding, author contributions, conflict-of-interest statements, and reference lists or author lists that leaked into the text. It stays searchable with `--regex`, but it is not embedded. |

### What gets stored

The `chunks` table has one row per chunk: `bibcode`, `arxiv_id`, `chunk_index` (the order within the paper), `section_heading`, `heading_path`, `kind`, `text` and `n_tokens`. A paper comes to about 30 chunks, and in the corpus this was developed on about 90% of them are prose.

```
pixi run chunk --dry-run                 # how many papers would be chunked, without writing anything
pixi run chunk --limit 5                 # chunk the first 5 pending papers, as a smoke test
pixi run chunk                           # every paper that has Markdown and no chunks yet
pixi run chunk --rebuild                 # drop the chunks table and chunk every paper again
```

- **Resumable:** a paper that already has chunks is skipped, and a paper without Markdown yet waits for `fetch` and `convert`. A paper that gains an arXiv ID through `fetch --refresh` is re-chunked from its new Markdown automatically.
- **Changing the chunk size:** edit `MAX_CHUNK_TOKENS` in `config.py`, run `chunk --rebuild`, and then run `embed` straight away. Vectors are stored apart from the chunks, together with a hash of the text they came from. So `--rebuild` keeps the vector of every chunk whose text did not change and only the changed chunks are embedded again. Until `embed` has run, search can return a changed chunk with the vector of its old text.
- **Re-converted papers:** a paper is chunked once. If you convert it again and get new Markdown, `chunk` does not notice, so use `--rebuild`.

## Stage 5: embed

`embed` turns every chunk into a vector, a list of 1,024 numbers that captures what the passage is about. Search compares your question with these vectors, so it finds passages by meaning and not only by matching words. The model runs on your own machine, on the GPU of your Mac through MLX. By default it is Microsoft's [harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) (MIT license). The first run downloads its weights, about 1.2 GB.

**This is slow.** The first embedding of a corpus of about 11,000 papers, 345,000 chunks and 200 million tokens took about 16 hours on the machine this was developed on, at roughly 3,500 tokens per second. Yours will differ. It only has to be done once: after that, adding papers only embeds the new chunks. It is safe to leave running overnight and to stop whenever you want.

**What gets embedded.** For each chunk it embeds the paper title, the heading path and the chunk text, so a passage under a generic heading still carries its context. For a table chunk, only the caption and header row. Boilerplate chunks (acknowledgements and the like) are skipped.

### What gets stored

The `chunk_embeddings` table has one row per chunk and model: `bibcode`, `chunk_index`, `model`, `text_hash` and `embedding`. The hash covers the exact text that was embedded, and the model is stored as its Hugging Face name. For this corpus the table adds about 1.4 GB to the database.

```
pixi run embed --dry-run                 # how many chunks are pending and roughly how long they would take, without embedding
pixi run embed --limit 200               # embed the first 200 pending chunks, as a smoke test
pixi run embed                           # embed everything that is pending
```

- **Resumable:** vectors are saved in batches of about 250,000 tokens. Ctrl-C loses at most the batch in progress, and the next run carries on.
- **Only changes are re-embedded:** a chunk is pending when it has no vector or when the hash of its text has changed. After `chunk --rebuild`, or after switching `EMBED_TITLE_PREFIX` in `config.py`, only the affected chunks are embedded again. Vectors for chunks that no longer exist are deleted at the start of a run.
- **The database is locked while it runs:** `embed` keeps the database open for writing until it finishes, so `search` and `browse` cannot open it in the meantime. For the first big run, that means search is unavailable for many hours.
- **Changing the model:** set `EMBED_MODEL` in `config.py` to another embedding model, and set `QUERY_INSTRUCTION` to the query format that model's model card asks for. The model must have a Qwen3 backbone and use last-token pooling, and `embed` stops with a clear message if it doesn't. Run `embed` again to embed every chunk with the new model. Search only uses the vectors of the configured model.

## Stage 6: search

`search` finds passages in the corpus and prints them with their ADS bibcodes. It is read-only and does retrieval only: no language model is involved, and nothing leaves your machine. You, or the AI agent you give it to, read the passages and write the answer. There is no server to run.

```
scix-search search --semantic 'how are black hole masses of little red dots estimated'
```

`scix-search` is the script `bin/scix-search`. To run it from any directory, link it onto your `PATH`, for example `ln -s "$PWD/bin/scix-search" ~/.local/bin/`. Always put queries, regexes and bibcodes in single quotes, because bibcodes contain `&` and regexes contain `\` and `|`. It runs in the `embed` environment, so run `pixi install -e embed` first if you haven't.

### Two ways to search

You can search by meaning, by text, or both. `search` takes `--semantic` for the first and `--regex` for the second, and needs at least one of them.

**Semantic** (`--semantic`) finds passages by meaning, even when they use different words from your question. Give it one to four phrasings of the same question: a plain question, a version with the field's terminology and abbreviations, and a sentence the way a paper would state it. The results are fused, so a passage that scores well on several phrasings rises. Each call loads the embedding model, so it takes a few seconds.

```
scix-search search --semantic 'how are black hole masses of little red dots estimated' \
                   --semantic 'virial black hole mass estimates for LRDs from broad H-alpha' \
                   --semantic 'Black hole masses of LRDs are inferred from the width of the broad Halpha line'
```

**Literal** (`--regex`) finds every passage whose text matches a regular expression. It is exhaustive and unranked: it lists all the matches and the papers they belong to, and it also searches the boilerplate sections that semantic search leaves out. No model is loaded, so it is fast. Use it for complete lists of papers, to check an exact number, name or quote, and to test a regex before you combine it with a semantic query.

```
scix-search search --regex 'Ca II triplet' --since 2024
scix-search search --regex 'CaT' --case --word --also 'little red dot'
```

The regex is case-insensitive and uses RE2 syntax (no lookahead or backreferences). Repeat `--regex` to allow several wordings, which are combined with OR. The text contains LaTeX, so allow `\W*` between words and use `\b` around short tokens. Special characters such as `.`, `(` and `+` need a backslash.

### Combining them

Pass both, and `search` runs the semantic search and also ranks the passages that match your regex. Matches rise in the ranking, are shown in bold in context, and every *other* paper that matches is listed afterwards. One call answers both "what do the papers say" and "which papers mention it", which suits named things: an object, a spectral line, an acronym, a number.

```
scix-search search --semantic 'which little red dots show Balmer absorption' \
                   --regex 'balmer\W+(break|absorption)'
```

The header says how many chunks and papers match the regex. Zero means the search fell back to semantic only, and thousands means the regex is too generic to help. `--papers 25` sets how many other matching papers are listed (0 turns the list off). For conceptual questions, leave the regex out.

### Options

| Option | Effect |
| --- | --- |
| `--since 2024` | Only papers from that year on. |
| `--paper BIBCODE` | Only this paper. |
| `--also REGEX` | Only papers that also match this regex somewhere. |
| `--case`, `--word` | Make `--regex` case-sensitive, or match whole words. |
| `--k N` | How many passages to show. The default is 8 with `--semantic` and 30 with `--regex` alone. |
| `--per-paper N` | At most this many passages from one paper. The default is 2 and 3. |
| `--papers N` | How many matching papers to list. |
| `--max-chars N` | How much of each passage to show. |

### Reading the results

Each passage starts with a line like `` `2026arXiv260922623W#1` arXiv:2609.22623 | 0.72 | q1,2 +regex | prose ``. That is the chunk's id (`bibcode#chunk number`), its arXiv ID, its cosine score, the phrasings that found it, `+regex` if it matches your regex, and its kind. Previews are clipped, and a `[+1860 chars]` marker says how much was cut. The score is only a ranking. A semantic search always returns something, so judge relevance by reading the passage, not by the number.

To look closer:

```
scix-search chunk '2026arXiv260922623W#1' --context 1     # the full passage and its neighbours
scix-search paper '2026arXiv260922623W'                   # metadata, abstract and an outline with chunk ranges
scix-search paper '2026arXiv260922623W' --chunks 3-5      # read a section
scix-search status                                         # what is indexed, embedded and where the corpus came from
```

### Using it from an AI agent

The tool is built for agents. It only reads, it never writes to the database or calls an outside service, and its output is plain text that fits in a model's context. An agent runs the same commands as above, and `skills/scix-search/SKILL.md` teaches it how to use them well: when to add a `--regex`, how to write it, to expand a passage before quoting a number, to judge relevance by content, and to cite as `First author et al. (year), bibcode` with the ADS link. It also tells the agent never to run the stages that write to the database.

- **Claude Code:** link the skill into your skills folder, `ln -s "$PWD/skills/scix-search" ~/.claude/skills/scix-search`, and make sure `scix-search` is on your `PATH`.
- **Other agents:** point the agent at `skills/scix-search/SKILL.md` or paste its contents into your agent's instructions file, and make sure it can run `scix-search`.
- **Describe your corpus:** the `description` at the top of `SKILL.md` is what tells an agent when to use the tool. It says to run `status` to see the corpus's scope, but you will get better results if you add your topics to it.

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

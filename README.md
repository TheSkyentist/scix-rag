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

The request count includes the clean-up pass at the end (see `--no-canonicalize` below). The numbers above are illustrative.

### Other options

| Option | Effect |
| --- | --- |
| `--max-seeds N` | Most seeds a `--query` may use (default 2000). A warning says how many matches were left out. |
| `--no-canonicalize` | Skip the clean-up pass at the end that merges papers ADS has re-keyed, such as a preprint later published under a new bibcode. |

### Good to know

- Re-running is cheap and safe. Seeds already in the database are refreshed, only new neighbors cost an ADS request, and you can add as many different queries as you like to one corpus.
- Each paper records how it got in (`query_source`): as a seed, or as a reference or citation of one. The first entry is kept when a later query finds the same paper.
- ADS allows about 5000 requests per day per key. A large expansion uses a few hundred. If you hit the limit, the run stops with a message and you can re-run it the next day.

## License

GPL-3.0. See [LICENSE](LICENSE).

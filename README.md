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
- **An ADS API key.** It is free; you can create one in your [ADS/SciX](https://scixplorer.org/user/settings/token) account settings.

## Disclaimers

1. **The text is often from the preprint, not the published paper.** Full text is fetched as HTML from the arXiv preprint when one exists. It can differ from the final published version, so check the journal version before you quote or cite a number.
2. **Not every paper can be fetched.** Papers with no arXiv preprint need a publisher or ADS-hosted PDF, and many don't have one. Some of those are behind a paywall or block automated downloads. Not every PDF converts cleanly either. Papers that can't be fetched or converted are left out of the index, so a search finding nothing is not evidence that no paper says it.
3. **This is not a replacement for reading the papers.** Retrieval tells you where to look. It does not make you an expert in the literature, and it can miss relevant passages. Read the papers and do your own work.

## License

GPL-3.0. See [LICENSE](LICENSE).

# Retrieval eval (October 2026)

Archive of the evaluation that chose `harrier-oss-v1-0.6b` and kept the chunking as is. The write-up is `docs/retrieval-eval.md`; the full tables are in `results/` (`phase*_summary.md`, `phase3_validation.md`).

## Contents

| Path | What it is |
|---|---|
| `sets/setA.jsonl`, `setB.jsonl` | Synthetic questions: citation-context and passage questions |
| `sets/setC_questions.json`, `setC.jsonl`, `setC_review.md` | Hand-curated questions with graded gold lists |
| `sets/phase1_subset.json` | The 220-question subset used for expensive configurations |
| `sets/hyde_passages.jsonl` | The hypothetical answers used for the HyDE test |
| `sets/review_sample.md` | Spot-check sample of Sets A and B |
| `search_gold.json` | The original 6 hand-curated cases, which also feed Set C |
| `mine_citation_contexts.py`, `sample_passages.py`, `build_sets.py` | Build Sets A and B; questions are written by Claude subagents in between |
| `pool_setc.py`, `build_setc.py` | Pool candidates for Set C and merge the judgments |
| `run_retrieval_eval.py` | Runner: numpy dense index, ranking cache, paired bootstrap / permutation / Holm |
| `configs.py` | Phase 1 configurations: BM25, HyDE, reranker blends |
| `rerank.py` | Qwen3-Reranker on MLX |
| `subset.py`, `phase2.py`, `embedders.py` | Phase 2 sub-corpus, chunk / prefix / model variants, and per-model loaders |
| `run_search_eval.py` | The original gold-set check for `search --term` |

## Re-running

The scratch artefacts were deleted after the decision:
- BM25 index (`data/eval_fts.duckdb`)
- sub-corpus (`data/eval_subset*.duckdb`)
- Phase 2 vectors and reranker scores (`data/eval_cache/`)
- ranking cache (`results/cache/`)
- the Qwen3, Jina and reranker weights
- the Qwen3-0.6B vectors in `papers.duckdb`

`run_retrieval_eval.py --sets A,B,C --configs dense,dense+term` works against whatever model `papers.duckdb` holds. Everything else must be rebuilt first: see each script's docstring, and re-add the model to `config.EMBED_MODELS` where needed.

The gold sets use this corpus's bibcodes as of October 2026. Run `canonicalize` before re-running, so re-keyed papers still match. It updates `search_gold.json` only.

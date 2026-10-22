# Retrieval evaluation (October 2026)

This page summarises a three-phase evaluation of the search stack. It decided the current embedding model (`harrier-oss-v1-0.6b`) and confirmed the chunking.

The eval code, question sets and full result tables live on the `retrieval-eval-2026-10` branch (`evals/`). They are tied to this particular corpus (bibcodes), so they are not on main.

## Eval sets

| Set | Questions | What counts as a correct answer | How it was built |
|---|---|---|---|
| A | 292 | The paper cited for a claim (the citing paper is excluded) | Sentences in the corpus that cite one in-corpus paper by author and year, rewritten as questions by Claude, with the authors and year removed. Questions that restated the cited title were dropped. |
| B | 284 | One specific chunk | Claude wrote one question per sampled chunk, stratified by era; about 10% are tables. |
| C | 20 | Graded answer lists, 20–50 papers each | 14 real research questions from the corpus owner plus the 6 original hand-curated cases. Candidates were pooled from dense search, regex matches and Claude's own searches, then graded by Claude. |

## Metrics and statistics

- Paper-level recall at k: found / min(k, number of correct papers). Also paper-level nDCG@10 and chunk-level MRR (Set B only).
- Configurations are compared per question: 95% bootstrap confidence intervals on the difference, sign-flip permutation tests, Holm-corrected for multiple comparisons.
- Recall@50 was close to its ceiling (~0.93 on A+B), so decisions used R@10, nDCG@10 and chunk MRR.
- Twin rows (the same paper stored as both a preprint and a journal record) are scored as one paper.

## Phase 1: retrieval stack, no re-index (Qwen3-Embedding-0.6B)

| Change | Verdict |
|---|---|
| Query instruction wording | No effect; kept as is |
| BM25 (DuckDB `fts`) on its own | Much worse than dense |
| Dense + BM25, RRF weight 0.3 | +0.08 R@10 on Set C without a regex, but it costs chunk MRR. With harrier it adds nothing over `--term`, so it was **not adopted**. |
| Regex `--term` | The strongest signal on Set C with Qwen (nDCG@10 +0.085) |
| Paper-level title+abstract vectors | Help "which paper" questions (A), hurt passage and topic questions; **dropped** |
| HyDE (an LLM-written hypothetical answer added as an extra query) | No effect; **dropped** |
| Qwen3-Reranker-0.6B over the top 100 | Chunk MRR +0.12 on B, but worse on C. Blending its scores fixed that, yet it adds nothing on top of `--term`, at ~15 s per query; **dropped**. The 4B reranker was not pursued. |

## Phase 2: re-embedding variants (1,012-paper sub-corpus, 220 questions)

| Variant | A+B nDCG@10 | B chunk MRR | C nDCG@10 |
|---|---|---|---|
| Qwen3-0.6B, title + heading prefix (old) | 0.723 | 0.509 | 0.645 |
| + abstract opening in the prefix | 0.752 | 0.504 | 0.652 |
| heading only (no title) | 0.639 | 0.411 | 0.590 |
| prev512 (causal stand-in for late chunking) | 0.728 | **0.230** | 0.638 |
| 512-token chunks | 0.719 | 0.470 | 0.629 |
| 2048-token chunks | 0.722 | 0.460 | 0.661 |
| **harrier-oss-v1-0.6b** | **0.769** | **0.576** | **0.709** |
| jina-embeddings-v5-text-small | 0.768 | 0.564 | 0.647 |

Conclusions:
- **Keep 1024-token, heading-aware chunks with the title + heading-path prefix.**
- Late chunking doesn't fit last-token-pooled decoder embedders. The causal stand-in wrecks chunk MRR because neighbouring chunks share the same context.
- Adding the abstract opening to the prefix helps A but costs recall on C (with harrier: C R@50 −0.066), so it was not adopted.
- **Model: harrier.** It has the same size and speed as Qwen3-0.6B. Jina does no better on C and is licensed non-commercial.
- Qwen3-Embedding-4B was not run: a full re-index would take ~150 h at ~385 tok/s.

## Phase 3: full-corpus validation (596 questions)

| | A+B nDCG@10 | B chunk MRR | C nDCG@10 | C nDCG@10 with `--term` |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 0.686 | 0.485 | 0.643 | 0.728 |
| **harrier-oss-v1-0.6b** | **0.731*** | **0.548*** | **0.698** | 0.701 |
| harrier + BM25@0.3 | 0.717 | 0.476 | 0.693 | 0.703 |

`*` = significant after Holm correction. Set C has only 20 questions, so its differences are not significant.

- With `--term`, harrier is roughly level with Qwen (−0.027, p ≈ 0.3). Harrier's dense ranking already holds most of the term-matching chunks, so term fusion moves it less; raising the term weight changed nothing.
- `--term` stays because of its coverage listing and highlighted matches.

**Cost:** re-embedding the corpus with harrier took 16.4 h at ~3,550 tok/s on an M2 Pro (16 GB).

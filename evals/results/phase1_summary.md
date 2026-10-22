# Phase 1 summary: retrieval-stack changes without re-indexing (2026-10-19)

Model: Qwen3-Embedding-0.6B, current chunks. Sets: A = 292 citation-context questions (gold = cited paper), B = 284 passage questions (gold = one chunk), C = 20 hand-curated questions (14 user questions with graded multi-paper gold + the 6 original `search_gold.json` cases). Expensive configs ran on the Phase 1 subset (100 A + 100 B + all of C). The adoption rule uses R@10, nDCG@10 and chunk MRR (R@50 is at ceiling, ~0.93 on A+B). Detailed tables:

- `phase1_cheap.md`: query instruction, BM25 (all items)
- `phase1_fusion.md`: weighted BM25, paper vectors (all items)
- `phase1_rerank06.md`: HyDE and the 0.6B reranker (subset)
- `phase1_rerank_blend.md`: reranker aggregation and blending (subset)
- `phase1_rerank_term.md`: reranker on top of regex + BM25 candidates (Set C)

| Change | A (which paper) | B (exact passage) | C (user topic questions) | Latency | Verdict |
|---|---|---|---|---|---|
| Query instruction (generic / evidence wording) | ±0.01 | MRR +0.03 (generic) | ±0.02 | – | **Keep current** |
| BM25 alone | nDCG −0.21 | nDCG −0.12 | −0.07 | 0.3 s | No |
| Dense + BM25 (weight 1) | ≈ | MRR −0.085* | ≈ | 0.3 s | No |
| **Dense + BM25 (weight 0.3)** | ≈ | MRR −0.04* | **R@10 +0.08\*** | 0.3 s | **Adopt as the default for searches without `--term`** |
| Paper vectors (title + abstract), fused | **nDCG +0.047\*** | nDCG −0.12* | −0.04 | 0.1 s | Optional mode for "which paper" questions only |
| HyDE (Claude-written hypothetical answer as an extra query) | ≈ | ≈ | ≈ | 0.2 s + LLM | **Drop** |
| Reranker 0.6B, replacing the fused order | ≈ | **MRR +0.12\*, nDCG +0.09\*** | nDCG −0.08 | ~15 s per variant | Hurts topic questions |
| Reranker 0.6B, max over variants + rank fusion (weight 1) | ≈ | MRR +0.05, R@5 +0.09* | nDCG +0.05 (but < dense+term) | ~15 s per variant | Optional "precise passage" mode |
| Regex `--term` (production) | n/a | n/a | **nDCG +0.085\*** (best on C) | 1–4 s | **Keep** |
| `--term` + BM25 0.3 | n/a | n/a | ≈ `--term` | – | No gain over `--term` |
| Reranker on top of `--term` + BM25 | n/a | n/a | nDCG −0.03 vs `--term` | ~12 s | No |

`*` = Holm-adjusted permutation p < 0.05. Set C has only 20 items, so its intervals are wide.

## Takeaways

1. On multi-paper topic questions, the regex term that Claude writes is still the strongest single signal. BM25 at weight 0.3 recovers most of it when no term is given.
2. The reranker is valuable only for pinpointing a single passage (Set B), and it costs about 15 s per query variant. Blended with the fused rank it no longer hurts topic questions, but it doesn't help them either once `--term` is in use. Per the agreed rule, the 4B reranker was not run.
3. Paper-level vectors help "find the paper behind this claim" questions and hurt everything else, so they should be a mode, not a default.
4. Nothing here needs a re-index. The open questions (model, chunk size, prefix, late chunking) are Phase 2.

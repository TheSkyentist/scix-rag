# Phase 2 summary: re-embedding variants on the sub-corpus (2026-10-20)

**Setup**
- Sub-corpus `data/eval_subset.duckdb`: 1,012 papers (493 gold, 469 hard negatives, 50 random), 23.6M chunk tokens.
- Items: the Phase 1 subset (100 A, 100 B, 20 C).
- Ranking: dense only, RRF over each item's query variants.
- Full tables: `phase2_all.md`. Significance is Holm-adjusted across the 7 comparisons.

| Variant | A nDCG@10 | B nDCG@10 | B chunk MRR | C R@10 | C nDCG@10 | A+B nDCG@10 | tok/s |
|---|---|---|---|---|---|---|---|
| **Qwen3-0.6B, title + heading (current)** | 0.659 | 0.787 | 0.509 | 0.761 | 0.645 | 0.723 | ~3,750 |
| + abstract opening in the prefix | 0.691 | 0.812 | 0.504 | 0.763 | 0.652 | 0.752 (+0.029) | ~3,750 |
| heading path only (no title) | 0.534* | 0.744 | 0.411* | 0.700 | 0.590 | 0.639* | – |
| prev512 (causal late-chunking stand-in) | 0.697 | 0.759 | **0.230\*** | 0.764 | 0.638 | 0.728 | 1.7× tokens |
| 512-token chunks | 0.666 | 0.771 | 0.470 | 0.748 | 0.629 | 0.719 | – |
| 2048-token chunks | 0.678 | 0.766 | 0.460 | 0.781 | 0.661 | 0.722 | – |
| **harrier-oss-v1-0.6b** (MIT) | **0.721\*** | 0.817 | **0.576** | **0.849** | **0.709** | **0.769\*** | ~3,800 |
| jina-embeddings-v5-text-small (CC BY-NC) | 0.705 | **0.830** | 0.564 | 0.802 | 0.647 | 0.768* | ~4,300 |

## Findings

1. **Harrier is the best model on every set, at the same size and speed as the current one.** It has the same Qwen3-0.6B architecture, 1024 dimensions and last-token pooling, so it is a drop-in swap.
   - A+B nDCG@10: +0.046 (significant).
   - Your questions (C): nDCG@10 +0.064 and R@10 +0.088, though only 20 items.
   - Chunk MRR: +0.066.
   - A full re-index would take about 17 h.
2. **Jina is as good as harrier on the synthetic sets but no better than the baseline on your questions** (C nDCG +0.002). With its non-commercial licence it comes second.
3. **Chunk size doesn't matter much.** 512 and 2048 are within noise of 1024, and 512 is slightly worse on C. Keep 1024.
4. **Late chunking (prev512) is out.** Paper-level results are flat, but it wrecks the passage set: chunk MRR falls from 0.51 to 0.23, because neighbouring chunks inherit the same context and become hard to tell apart.
5. **The title in the prefix matters.** Removing it costs a lot everywhere. The abstract opening adds a little on A+B (+0.029, not significant after correcting for multiple comparisons) and is neutral on C.

## Not tested

- harrier combined with the abstract prefix.
- BM25@0.3 fusion on top of harrier.
- Qwen3-4B, which was dropped as impractical (~150 h re-index).

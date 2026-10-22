---
name: scix-search
description: Search a local full-text index of astronomy papers from NASA ADS (built with scix-rag) and get passages back with ADS bibcodes. Use whenever the user asks what the literature says about the topics the corpus covers (run `status` to see what it was seeded from), wants evidence, numbers, quotes or citations from papers, asks which papers discuss or mention something, or asks to find, check or verify a claim in the literature. Retrieval only: you read the passages and write the answer.
---

# scix-search: local literature search

A local index of passages from ADS papers (full text, converted to Markdown, split into chunks, embedded with a local model). There is no LLM and no network access involved: you do the reading and writing.

Command (it must be on your `PATH`; otherwise it is `bin/scix-search` in the scix-rag repo):

    scix-search <search|chunk|paper|status> ...

**Always single-quote bibcodes, queries and regexes** (bibcodes contain `&`, regexes contain `\` and `|`).

**First, run `scix-search status`.** It shows how many papers and passages are indexed and what the corpus was seeded from (the `search:<query>` lines). That is the scope of the corpus: say so when it limits an answer.

## The one command you usually need

    search --semantic 'phrasing 1' [--semantic 'phrasing 2'] [--regex 'REGEX'] [--k 8] [--papers 25] [--since YEAR]

At least one of `--semantic` and `--regex` is required, and the combination decides what happens:

- **`--semantic` only:** passages ranked by meaning. Give 1-4 phrasings of the same question; they are fused by reciprocal rank. The model loads once (about 3 s).
- **`--regex` only:** an exhaustive, unranked match over all chunk text, boilerplate sections included. No model is loaded, so it is fast. It lists every paper with a match, with snippets. Use it for (a) the complete list of papers that mention something, (b) checking an exact number, name or quote, (c) testing a regex before you pair it with `--semantic`.
- **Both:** the semantic ranking is also computed over only the passages that match the regex, and the two are fused, so on-topic passages that contain the term rise. Matches are shown in context with the match in **bold**, and every *other* matching paper is listed after the passages, so "which papers mention X?" is answered by the same call. With a good embedding model the semantic ranking usually has the matching passages near the top already, so the main gains are the highlighted matches and the coverage list.

Filters that work in every mode: `--since YEAR`, `--paper 'BIBCODE'` (one paper only), `--also 'REGEX'` (keep only papers that also mention this somewhere, e.g. `--also 'some corpus topic'`), `--case` (case-sensitive regex), `--word` (whole-word regex).

### When to add `--regex` to a semantic search

- **Yes**, when you need completeness or exact mentions of something specific: "which papers detect / discuss / use X", a spectral line or feature (`Ca II triplet`, `O I 8446`, `[Fe II]`), an object (`NGC 4151`), an instrument mode, an acronym, a number or unit, a formula. It costs about 2 s.
- **No**, for conceptual questions ("what explains the V-shaped continuum?"). Semantic search handles these well on its own.
- **Not as a fix for weak results.** If the passages are off-topic, rephrase the query; if you need every mention, use `--regex` alone.

### Writing the regex

Case-insensitive by default, RE2 syntax (no lookahead/lookbehind/backreferences). You know the field's spelling variants, so spell them out:

- Ca II triplet, any spelling: `--regex 'ca\W*(ii)?\W*triplet|calcium\W+triplet|\bCa\s?T\b'`
- O I 8446 (the text often has LaTeX, `O i $\lambda$ 8446`, so allow `\lambda`): `--regex 'O\s?i\W*(\\lambda\W*|λ\W*)?8446'`
- a named object: `--regex '\bNGC\s?4151\b'` (escape `+ . ( ) [ ]`)
- an acronym that clashes with a word or another acronym: `--regex 'CaT' --case --word` (plain `CaT` also matches "CAT" and "cat")
- several wordings: `--regex 'ionis(ed|ing)\W+outflow' --regex 'photoionis\w+\W+wind'` (repeated `--regex` flags are OR-ed)

Rules of thumb: allow `\W*` between words (spaces, hyphens, LaTeX braces), use `\b` for short or ambiguous tokens, give alternations for spellings (`Fe\s?(ii|II)`), and prefer 2-3 specific patterns to one vague one. Do **not** put the corpus's own topic in the regex or query unless that is the point: the whole index is about it, so the word adds noise.

### Reading the output

- `# regex matched N chunks in M papers`: how common the pattern is. **0 chunks** triggers a warning that the results are semantic-only: loosen the regex and test it with `--regex` alone. **More than 5000 chunks** triggers a "too generic" warning: narrow the regex.
- Per passage: `BIBCODE#INDEX | cosine | q1,q2 +regex | kind`. `q…` = which phrasings retrieved it, `+regex` = it matches the regex.
- `## Other papers matching the regex (N; listing M ...)`: papers whose text matches but whose passages did not make the top `--k`. A regex match is not relevance: skim titles, and expand a passage (`chunk`) before relying on one. Matches in unrelated papers (a different "CAT") are normal for an acronym.

### Sizing the output: `--papers N` and `--k`

With both signals, `--papers` is how many *other* matching papers to list (default 25, hard max 300, 0 = none; each line is about 50 tokens). With `--regex` alone it is the length of the papers summary (default 50).

- Focused question ("what do papers say about X"): leave the defaults.
- Survey question ("which papers discuss / detect / use X", "list everything on X"): raise `--papers`, e.g. `--papers 100`, and read the passages plus the list.
- If the header reports hundreds of matching papers, listing them all is the wrong move: narrow the regex, add `--since`, require a co-occurring topic with `--also`, or accept a sample and say so.
- `--k` is the number of passages shown: 8 by default with `--semantic`, 30 snippets with `--regex` alone. Keep it at 8-15 for semantic searches. If the passages are weak, change the phrasing or the regex rather than raising `--k`.

## Other commands

- `chunk 'BIBCODE#INDEX' [--context 1]`: the full passage plus neighbouring chunks.
- `paper 'BIBCODE'`: metadata, abstract and outline (headings with chunk ranges). Then `paper 'BIBCODE' --chunks A-B` to read a section. `--full` reads everything up to `--max-chars`; avoid unless really needed.
- `status`: what is indexed and embedded, and what the corpus was seeded from. Run it first if anything looks off.

## How to use it well

1. Decide whether you need coverage or exact mentions of a named thing. If so, plan a `--regex`; if not, plain `--semantic`.
2. Write 2-3 phrasings of the *same* question: a natural-language question, a terminology variant with synonyms and abbreviations, and a declarative sentence as a paper would state it. Do not put years or authors in the query; use `--since` for a year cutoff.
3. **Expand before you quote or cite a number.** Previews are clipped (`[+N chars]`) and keyword windows show only part of the passage. Read the whole passage with `chunk` first.
4. **Judge relevance by content, not by score.** Cosine scores are only a ranking: on-topic hits are typically 0.55-0.8, but generic passages (a title block, a table of measurements) can score above 0.6 on a query that merely mentions the corpus topic. Dense search always returns *something*; if the top passages are weak or off-topic, say so instead of stretching them.
5. Two-step for "tell me about SOURCE-X": `search` with `--semantic` for a descriptive phrasing and `--regex` for the name, then `chunk --context 1` on the hits (papers often say "this source" instead of the name).
6. If a result seems incomplete, check with `--regex` alone (exhaustive) before concluding the literature says nothing.

## Citing

Cite as `First author et al. (year), bibcode`, with `https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract` and, when the paper has an arXiv id, `https://arxiv.org/abs/{arxiv_id}`. Mention the chunk id (`BIBCODE#INDEX`) for any quoted passage and quote only text you have expanded. Citation counts are ADS values as of the corpus snapshot.

## Coverage limits (tell the user when they matter)

- The corpus is only what was ingested (see the seed queries in `status`), not all of astronomy. No hits is weak evidence that the literature says nothing.
- Only papers whose full text was fetched are included, and the text is often from the arXiv preprint rather than the published version. Reference lists, acknowledgement-style boilerplate and leaked author/affiliation blocks are not searchable by `--semantic` (`--regex` alone still sees them).
- Figures are captions only; equations are LaTeX. Tables are indexed by caption and header row only for ranking (the whole table appears when you expand it, and `--regex` matches the full table text).
- `--regex` patterns match the extracted Markdown text, so line names may appear with LaTeX (`$\lambda$ 8446`, `Ca ii`) or Unicode: allow for it.
- New papers are searchable only after the embed step has run (`status` shows coverage).
- Citation edges are a one-hop star, not a complete cited-by graph.

## When things go wrong

- `cannot open ... Conflicting lock` / "database locked": a pipeline job (usually `pixi run embed`, which can take hours) holds the database. Tell the user and retry later. Do not try to work around it.
- `embed environment missing`: tell the user to run `pixi install -e embed` in the scix-rag repo. Do not run it yourself.
- `no embeddings for ...`, or a low-coverage warning: run `status`, report the numbers, and stop.
- `--regex rejected`: RE2 syntax error (usually a lookaround or backreference); rewrite with `|`, `\W*`, `\b`.
- **Never run `embed`, `chunk`, `fetch`, `convert`, `query` or anything else that writes to the database or calls external APIs.** This tool is read-only.

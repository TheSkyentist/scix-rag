---
name: arxivrag-search
description: Search a local full-text index of ~10,800 astro-ph papers (Little Red Dots / LRDs, JWST compact broad-line AGN, black-hole seeds, and the papers they cite or are cited by, including older foundational work) and get passages back with ADS bibcodes. Use whenever the user asks what the literature says about little red dots, LRDs, JWST AGN or related topics, wants evidence, numbers, quotes or citations from papers, asks which papers discuss or mention something, or asks to find, check or verify a claim in the astro-ph literature, from any project. Retrieval only: you read the passages and write the answer.
---

# arxivrag-search: local astro-ph literature search

A local index of about 334K passages from about 10.8K papers (full text, converted to Markdown, split into chunks, embedded with Microsoft's harrier-oss-v1-0.6b). There is no LLM and no network access involved: you do the reading and writing.

Command (works from any directory):

    /Users/hviding/Projects/arXivRAG/bin/arxivrag-search <search|chunk|paper|grep|status> ...

**Always single-quote bibcodes, queries and regexes** (bibcodes contain `&`, regexes contain `\` and `|`).

## The one command you usually need

    search 'question phrasing 1' 'phrasing 2' --term 'REGEX' [--term 'REGEX2'] [--k 8] [--papers 25] [--since YEAR]

It does three things in one call (model loads once, ~3 s; the regex scan adds ~2 s):

1. **Semantic search** over all passages with your 1-4 phrasings, fused by reciprocal rank.
2. **Literal search**: if you pass `--term`, the same ranking is *also* computed over only the passages that match the regex, and the two are fused, so on-topic passages that contain the term are nudged up. With the current embedding model the semantic ranking usually has them near the top already, so expect this to reorder the passages only a little.
3. **Coverage**: after the passages, every *other* paper matching the term is listed (one line each, ordered by number of matching passages), so "which papers mention X?" is answered by the same call. This list, and the highlighted matches below, are the main reasons to pass `--term`.

Passages that match the term show the match in context with the match in **bold** (windows separated by `[…]`), so the relevant sentence is not clipped away.

### When to pass `--term`

- **Yes**, when you need completeness or exact mentions of something specific: "which papers detect / discuss / use X", a spectral line or feature (`Ca II triplet`, `O I 8446`, `[Fe II]`), an object (`The Egg`, `GN-28074`, `UNCOVER-45924`), an instrument mode, an acronym, a number or unit, a formula. It costs ~2 s.
- **No**, for conceptual questions ("what explains the V-shaped continuum?", "how could electron scattering broaden the lines?"). Semantic search handles these well on its own.
- **Not as a fix for weak results.** If the passages are off-topic, rephrase the query; if you need every mention, use `grep`.

### Writing the regex

Case-insensitive by default, RE2 syntax (no lookahead/lookbehind/backreferences). You know the field's spelling variants, so spell them out:

- Ca II triplet, any spelling: `--term 'ca\W*(ii)?\W*triplet|calcium\W+triplet|\bCa\s?T\b'`
- O I 8446 (the text often has LaTeX, `O i $\lambda$ 8446`, so allow `\lambda`): `--term 'O\s?i\W*(\\lambda\W*|λ\W*)?8446'`
- a named object: `--term '\bThe Egg\b|J1025\+1402'` (escape `+ . ( ) [ ]`)
- an acronym that clashes with a word or another acronym: `--term '(?-i:\bCaT\b)'` for exact case (plain `CaT` also matches "CAT", the Cosmic Archaeology Tool, and "cat")
- several wordings: `--term 'ionis(ed|ing)\W+outflow' --term 'photoionis\w+\W+wind'` (repeated `--term`s are OR-ed)

Rules of thumb: allow `\W*` between words (spaces, hyphens, LaTeX braces), use `\b` for short or ambiguous tokens, give alternations for spellings (`Fe\s?(ii|II)`), and prefer 2-3 specific terms to one vague one. Do **not** put the population in the regex or query ("LRD", "little red dot") unless that is the point: the whole index is about them, so the word adds noise and pulls in generic LRD passages.

### Reading the output

- `# term matched N chunks in M papers`: how common the term is. **0 chunks** triggers a warning that the results are semantic-only: loosen the regex and test it with `grep --regex`. **More than 5000 chunks** triggers a "too generic" warning: narrow the regex.
- Per passage: `BIBCODE#INDEX | cosine | q1,q2 +term | kind`. `q…` = which phrasings retrieved it, `+term` = it matches the regex.
- `## Other papers matching the term (N; listing M ...)`: papers whose text matches but whose passages did not make the top `--k`. A regex match is not relevance: skim titles, and expand a passage (`chunk`) before relying on one. Matches in unrelated papers (a different "CAT", a stellar-population paper) are normal for an acronym.

### Sizing the list: `--papers N` (and `--k`)

`--papers` is how many *other* matching papers to list (default 25, hard max 300, 0 = none). Each line is ~50 tokens.

- Focused question ("what do papers say about X"): leave the default.
- Survey question ("which papers discuss / detect / use X", "list everything on X"): raise it, e.g. `--papers 100`, and read the passages plus the list.
- If the header reports hundreds of matching papers, listing them all is the wrong move: narrow the regex, add `--since`, require a co-occurring term with `grep --also`, or accept a sample and say so.
- Keep `--k` at 8-15. If the passages are weak, change the phrasing or the term rather than raising `--k`.

## Other commands

- `chunk 'BIBCODE#INDEX' [--context 1]`: the full passage plus neighbouring chunks.
- `paper 'BIBCODE'`: metadata, abstract and outline (headings with chunk ranges). Then `paper 'BIBCODE' --chunks A-B` to read a section. `--full` reads everything up to `--max-chars`; avoid unless really needed.
- `grep 'text' [--regex] [--word] [--case] [--since YEAR] [--also 'REGEX'] [--paper 'BIBCODE'] [--papers 50] [--limit 30]`: the **exhaustive, unranked** literal/regex match over all chunk text, including boilerplate sections. Use it to (a) get the complete list of papers that mention something, (b) check an exact number, name or quote, (c) test a regex before using it as `--term`. `--word` wraps the pattern in `\b…\b`; `--case` makes it case-sensitive; `--also 'little red dot'` keeps only papers that also mention that anywhere (e.g. `grep 'CaT' --case --word --also 'little red dot' --since 2023`). Without `--regex` the pattern is literal.
- `status`: what is indexed and embedded. Run it first if anything looks off.

## How to use it well

1. Decide whether you need coverage or exact mentions of a named thing. If so, plan a `--term`; if not, plain `search`.
2. Write 2-3 phrasings of the *same* question: a natural-language question, a terminology variant with synonyms and abbreviations, and a declarative sentence as a paper would state it. Do not put years or authors in the query; use `--since` for a year cutoff.
3. **Expand before you quote or cite a number.** Previews are clipped (`[+N chars]`) and keyword windows show only part of the passage. Read the whole passage with `chunk` first.
4. **Judge relevance by content, not by score.** Cosine scores are only a ranking: on-topic hits are typically 0.55-0.8, but generic passages (a title block, a table of measurements) can score above 0.6 on a query that merely mentions the population. Dense search always returns *something*; if the top passages are weak or off-topic, say so instead of stretching them.
5. Two-step for "tell me about SOURCE-X": `search` with `--term` for the name and a descriptive phrasing, then `chunk --context 1` on the hits (papers often say "this source" instead of the name).
6. If a result seems incomplete, check with `grep` (exhaustive) before concluding the literature says nothing.

## Citing

Cite as `First author et al. (year), bibcode`, with `https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract` and, when the paper has an arXiv id, `https://arxiv.org/abs/{arxiv_id}`. Mention the chunk id (`BIBCODE#INDEX`) for any quoted passage and quote only text you have expanded. Citation counts are ADS values as of the corpus snapshot.

## Coverage limits (tell the user when they matter)

- The corpus is the citation-graph neighbourhood of Little Red Dots seed papers (papers citing them and, for the seeds, papers they cite), not all of astro-ph. No hits is weak evidence that the literature says nothing.
- Only papers whose full text was fetched are included. Reference lists, acknowledgement-style boilerplate and leaked author/affiliation blocks are not searchable by `search` (`grep` still sees them).
- Figures are captions only; equations are LaTeX. Tables are indexed by caption and header row only for ranking (the whole table appears when you expand it, and `--term`/`grep` match the full table text); some papers converted before a converter fix lack list items and display equations.
- `--term` regexes match the extracted Markdown text, so line names may appear with LaTeX (`$\lambda$ 8446`, `Ca ii`) or Unicode: allow for it.
- New papers are searchable only after the embed step has run (`status` shows coverage).
- Citation edges are a one-hop star, not a complete cited-by graph.

## When things go wrong

- `cannot open ... Conflicting lock` / "database locked": a pipeline job (usually `pixi run embed`, which can take hours) holds the database. Tell the user and retry later. Do not try to work around it.
- `embed environment missing`: tell the user to run `pixi install -e embed` in the arXivRAG repo. Do not run it yourself.
- `no embeddings for harrier-oss-v1-0.6b`, or a low-coverage warning: run `status`, report the numbers, and stop.
- `--term regex rejected`: RE2 syntax error (usually a lookaround or backreference); rewrite with `|`, `\W*`, `\b`.
- **Never run `embed`, `chunk`, `fetch`, `convert`, `query` or anything else that writes to the database or calls external APIs.** This tool is read-only.

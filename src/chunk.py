"""
Split each paper's converted Markdown (data/markdown/*.md) into passage-
sized chunks for embedding. Pure text transform -- no model inference, no
embedding computed here (see embed.py). Reads data/markdown/*.md and
papers.duckdb (read-only), writes only to the `chunks` table.

Chunk boundaries follow the Markdown's own heading structure (already
clean from html_to_markdown.py's `#`/`##`/`###`): paragraphs are packed
into each section up to config.MAX_CHUNK_TOKENS, cut only at paragraph
boundaries, no overlap between adjacent chunks. Each chunk records its
innermost heading (`section_heading`) and the full path of enclosing
headings (`heading_path`, e.g. "2 Method > 2.3 Raman Scattering > 2.3.1
Physical Mechanism"; the document title is left out -- the embed stage
prepends it separately) so a generic sub-heading like "Monte Carlo
Simulation" keeps its parent's context.

Every chunk has a `kind`:
  prose        ordinary text (the default)
  table        one whole table (caption + rows), or an over-budget numeric/LaTeX
               data dump, never packed with prose; stored in full but only
               table_embed_text() -- caption + header row -- is meant to be
               embedded. A section's tables are emitted after its prose chunks
               so they don't fragment the prose.
  boilerplate  acknowledgements, data-availability, author-contribution,
               conflict-of-interest, funding, (leaked) reference sections and
               leaked front matter (author lists, affiliations, ORCIDs, keywords).
               Kept so they stay searchable on request, but the embed stage
               can skip them by default.

A paragraph longer than the budget is split further instead of becoming one
oversized chunk: at newlines, then at "• " bullets (the HTML converter used
to flatten lists into one "•"-joined line), then at sentence boundaries, and
the pieces are re-packed up to the budget. A single sentence that alone
exceeds the budget still becomes its own oversized chunk.

Token counts use Qwen3-Embedding-4B's own tokenizer (shared by the 0.6B
model), not a proxy -- loading just the tokenizer needs no GPU and no model
weights.

A paper is skipped if it already has any rows in `chunks`. Re-chunking after
a logic change means `--rebuild`, which drops the whole table first (and so
any embeddings stored in it once the embed stage exists).

Run directly (from the `chunk` environment):
    pixi run chunk
    pixi run chunk --limit 5   # smoke-test on a handful first
    pixi run chunk --rebuild   # drop `chunks` and re-chunk everything
"""

from __future__ import annotations

import re
from typing import Callable, NamedTuple

import duckdb
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import config

_TOKENIZER_NAME = "Qwen/Qwen3-Embedding-4B"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    bibcode         VARCHAR,
    arxiv_id        VARCHAR,
    chunk_index     INTEGER,
    section_heading VARCHAR,
    heading_path    VARCHAR,
    kind            VARCHAR,
    text            VARCHAR,
    n_tokens        INTEGER,
    PRIMARY KEY (bibcode, chunk_index)
);
"""

_INSERT_CHUNK = """
INSERT INTO chunks (bibcode, arxiv_id, chunk_index, section_heading, heading_path, kind, text, n_tokens)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Leading section numbering ("5.3 ", "VIII ", "A. ") -- stripped only to classify.
_NUMBERING_RE = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*\.?|[IVXL]+\.?|[A-Z]\.)\s+")
_BOILERPLATE_RE = re.compile(
    r"(acknowledg|data availab|availability of|code availab|author contribution|"
    r"competing interest|conflicts? of interest|disclosure|declaration of|funding|"
    r"bibliography|literature cited|works cited)|(references?|reference list)\s*$|"
    # Front matter that leaked into the body (whole-heading matches only, so
    # "Comparison to other authors" and the like stay prose).
    r"(affiliations?|authors?|authors?[’'] affiliations?|authors? and affiliations?|"
    r"author information|author list|complete author list( and affiliations?)?|"
    r"corresponding authors?|orcid( id)?s?|keywords?|special capabilities needed)"
    r"[\s:.]*$"
)

_TABLE_LINE_MIN_PIPES = 2
# An over-budget block this dense (prose is ~3.8 chars/token) is a flattened
# numeric/LaTeX data dump (e.g. a MinerU deluxetable), not text worth splitting.
_DENSE_CHARS_PER_TOKEN = 2.2
_SEPARATOR_ROW_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")

_BULLET_SPLIT_RE = re.compile(r"\s+(?=•\s)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\"“(\[])")
_ABBREVIATIONS = {
    "al", "fig", "figs", "eq", "eqs", "sec", "sect", "secs", "ref", "refs",
    "tab", "cf", "vs", "e.g", "i.e", "approx", "no", "nos", "vol", "ca", "resp",
}


class Chunk(NamedTuple):
    heading: str | None
    heading_path: str | None
    kind: str
    text: str
    n_tokens: int


class Section(NamedTuple):
    heading: str | None
    trail: tuple[str, ...]  # enclosing headings, outermost first, excluding the document title
    paragraphs: list[str]


def _split_sections(markdown: str) -> list[Section]:
    """Split into sections in document order. Blocks are blank-line-separated,
    matching html_to_markdown.convert()'s own "\\n\\n".join(...) output."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    heading: str | None = None
    trail: tuple[str, ...] = ()
    paragraphs: list[str] = []
    seen_heading = False
    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        m = _HEADING_RE.match(block)
        if m:
            if paragraphs:
                sections.append(Section(heading, trail, paragraphs))
            level, heading = len(m.group(1)), m.group(2).strip()
            is_title = not seen_heading and level == 1
            seen_heading = True
            if not is_title:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
            trail = tuple(h for _, h in stack if h)
            paragraphs = []
        else:
            paragraphs.append(block)
    if paragraphs:
        sections.append(Section(heading, trail, paragraphs))
    return sections


def _is_boilerplate(section: Section) -> bool:
    headings = section.trail + ((section.heading,) if section.heading else ())
    return any(
        _BOILERPLATE_RE.match(_NUMBERING_RE.sub("", h).lower()) for h in headings
    )


def _is_table(block: str) -> bool:
    """HTML path: "**Table.** caption" + pipe-delimited rows (or bare rows for
    an uncaptioned table). PDF path (MinerU): markdown pipe tables or raw
    <table> HTML."""
    if block.startswith("**Table.**") or "<table" in block:
        return True
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) < 3:
        return False
    tabular = sum(ln.count("|") >= _TABLE_LINE_MIN_PIPES for ln in lines)
    return tabular >= 3 and tabular / len(lines) >= 0.7


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for piece in _SENTENCE_SPLIT_RE.split(text):
        if out:
            last = out[-1].rsplit(None, 1)[-1].rstrip(".").lower()
            # An abbreviation or an author initial ("et al. Smith", "J. Smith"):
            # not a real sentence end.
            if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
                out[-1] += " " + piece
                continue
        out.append(piece)
    return out


# (separator rejoining the pieces, splitter), coarsest first.
_SPLITTERS: list[tuple[str, Callable[[str], list[str]]]] = [
    ("\n", lambda t: t.split("\n")),
    ("\n", lambda t: _BULLET_SPLIT_RE.split(t) if t.count("• ") >= 2 else [t]),
    (" ", _split_sentences),
]


def _split_oversized(
    text: str, count: Callable[[str], int], max_tokens: int, level: int = 0, sep: str = ""
) -> list[tuple[str, str]]:
    """Break text into (piece, separator-before-it) units that each fit
    max_tokens where possible, trying each splitter in turn only on pieces
    that are still too long. A piece no splitter can shorten is kept whole."""
    if level == len(_SPLITTERS) or count(text) <= max_tokens:
        return [(text, sep)]
    piece_sep, splitter = _SPLITTERS[level]
    parts = [p for p in splitter(text) if p.strip()]
    if len(parts) < 2:
        return _split_oversized(text, count, max_tokens, level + 1, sep)
    units: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        units.extend(_split_oversized(part, count, max_tokens, level + 1, sep if i == 0 else piece_sep))
    return units


def _pack_units(
    units: list[tuple[str, str]], count: Callable[[str], int], max_tokens: int
) -> list[tuple[str, int]]:
    packed: list[tuple[str, int]] = []
    buf, buf_tokens = "", 0
    for text, sep in units:
        n = count(text)
        if buf and buf_tokens + n > max_tokens:
            packed.append((buf, buf_tokens))
            buf, buf_tokens = text, n
        else:
            buf = buf + sep + text if buf else text
            buf_tokens += n
    if buf:
        packed.append((buf, buf_tokens))
    return packed


def _pack_section(
    section: Section, count: Callable[[str], int], max_tokens: int
) -> list[Chunk]:
    """Pack one section's paragraphs into <= max_tokens chunks, no overlap.
    Tables become their own chunks (after the prose); an oversized paragraph
    is split further rather than kept whole."""
    path = " > ".join(section.trail) or None
    prose_kind = "boilerplate" if _is_boilerplate(section) else "prose"

    chunks: list[Chunk] = []
    tables: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            chunks.append(Chunk(section.heading, path, prose_kind, "\n\n".join(buf), buf_tokens))
        buf, buf_tokens = [], 0

    for para in section.paragraphs:
        n = count(para)
        if _is_table(para) or (n > max_tokens and len(para) < _DENSE_CHARS_PER_TOKEN * n):
            tables.append((para, n))
            continue
        if n > max_tokens:
            flush()
            for text, n_tok in _pack_units(_split_oversized(para, count, max_tokens), count, max_tokens):
                chunks.append(Chunk(section.heading, path, prose_kind, text, n_tok))
            continue
        if buf and buf_tokens + n > max_tokens:
            flush()
        buf.append(para)
        buf_tokens += n
    flush()

    for table, n_tok in tables:
        chunks.append(Chunk(section.heading, path, "table", table, n_tok))
    return chunks


def chunk_paper(markdown: str, tokenizer: PreTrainedTokenizerBase) -> list[Chunk]:
    """Pure function: Markdown text -> chunks in document order."""

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return [
        c
        for section in _split_sections(markdown)
        for c in _pack_section(section, count, config.MAX_CHUNK_TOKENS)
    ]


def table_embed_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = config.MAX_TABLE_EMBED_TOKENS,
) -> str:
    """The part of a `kind='table'` chunk that gets embedded: caption plus
    header row, truncated to max_tokens. The embed stage calls this; the full
    table stays in `text`."""
    if "<table" in text:  # MinerU raw HTML: strip tags, one line
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    lines = [
        ln.strip()
        for ln in text.split("\n")
        if ln.strip() and not _SEPARATOR_ROW_RE.match(ln.strip())
    ]
    head = lines[:2] if lines and lines[0].startswith("**Table.**") else lines[:1]
    ids = tokenizer.encode("\n".join(head), add_special_tokens=False)[:max_tokens]
    return tokenizer.decode(ids)


def chunk(limit: int | None = None, rebuild: bool = False) -> None:
    con = duckdb.connect(str(config.DB_PATH))
    if rebuild:
        con.execute("DROP TABLE IF EXISTS chunks")
    con.execute(_SCHEMA)
    columns = {r[1] for r in con.execute("PRAGMA table_info('chunks')").fetchall()}
    if not {"kind", "heading_path"} <= columns:
        raise SystemExit(
            "chunks table has the old schema (no kind/heading_path); "
            "re-run with --rebuild to drop it and re-chunk everything"
        )

    already_chunked = {
        r[0] for r in con.execute("SELECT DISTINCT bibcode FROM chunks").fetchall()
    }
    # A paper that gained an arxiv_id after being chunked (fetch --refresh) now has a different
    # stem and source text; its old chunks are replaced once the new Markdown exists.
    stale = {
        r[0]
        for r in con.execute(
            """SELECT DISTINCT c.bibcode FROM chunks c JOIN papers p USING (bibcode)
               WHERE c.arxiv_id IS DISTINCT FROM p.arxiv_id"""
        ).fetchall()
    }
    all_papers = con.execute("SELECT bibcode, arxiv_id FROM papers").fetchall()
    todo = [
        (bibcode, arxiv_id)
        for bibcode, arxiv_id in all_papers
        if (bibcode not in already_chunked or bibcode in stale)
        and (config.MARKDOWN_DIR / f"{arxiv_id or bibcode}.md").exists()
    ]
    if limit:
        todo = todo[:limit]
    print(f"Chunking {len(todo)} papers")

    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

    for bibcode, arxiv_id in todo:
        stem = arxiv_id or bibcode
        markdown = (config.MARKDOWN_DIR / f"{stem}.md").read_text(encoding="utf-8")
        packed = chunk_paper(markdown, tokenizer)
        if bibcode in stale:
            con.execute("DELETE FROM chunks WHERE bibcode = ?", [bibcode])
        rows = [
            (bibcode, arxiv_id, i, c.heading, c.heading_path, c.kind, c.text, c.n_tokens)
            for i, c in enumerate(packed)
        ]
        con.executemany(_INSERT_CHUNK, rows)
        print(f"  [ok] {stem}: {len(rows)} chunks")

    con.commit()
    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk converted Markdown into passage-sized pieces")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending papers")
    parser.add_argument("--rebuild", action="store_true", help="Drop the chunks table and re-chunk every paper")
    args = parser.parse_args()

    chunk(limit=args.limit, rebuild=args.rebuild)

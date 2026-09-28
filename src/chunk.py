"""
Split each paper's converted Markdown (data/markdown/*.md) into passage-
sized chunks for embedding. Pure text transform -- no model inference, no
embedding computed here (see embed.py). Reads data/markdown/*.md and
papers.duckdb (read-only), writes only to the new `chunks` table.

Chunk boundaries follow the Markdown's own heading structure (already
clean from html_to_markdown.py's `#`/`##`/`###`): paragraphs are packed
into each section up to config.MAX_CHUNK_TOKENS, cut only at paragraph
boundaries, no overlap between adjacent chunks. A section longer than the
budget becomes multiple sequential chunks sharing the same heading; a
single paragraph that alone exceeds the budget becomes its own oversized
chunk rather than being split mid-paragraph.

Token counts use Qwen3-Embedding-4B's own tokenizer (the model embed.py
will use), not a proxy -- loading just the tokenizer needs no GPU and no
model weights.

A paper is skipped if it already has any rows in `chunks` (re-chunking
after a logic change means deleting its rows first, not something this
does automatically).

Run directly (from the `chunk` environment):
    pixi run chunk
    pixi run chunk --limit 5   # smoke-test on a handful first
"""

from __future__ import annotations

import re

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
    text            VARCHAR,
    n_tokens        INTEGER,
    PRIMARY KEY (bibcode, chunk_index)
);
"""

_INSERT_CHUNK = """
INSERT INTO chunks (bibcode, arxiv_id, chunk_index, section_heading, text, n_tokens)
VALUES (?, ?, ?, ?, ?, ?)
"""

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_sections(markdown: str) -> list[tuple[str | None, list[str]]]:
    """Split into (heading, paragraphs) segments in document order. Blocks
    are blank-line-separated, matching html_to_markdown.convert()'s own
    "\\n\\n".join(...) output."""
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    paragraphs: list[str] = []
    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        m = _HEADING_RE.match(block)
        if m:
            if paragraphs:
                sections.append((heading, paragraphs))
            heading = m.group(2).strip()
            paragraphs = []
        else:
            paragraphs.append(block)
    if paragraphs:
        sections.append((heading, paragraphs))
    return sections


def _pack_chunks(
    sections: list[tuple[str | None, list[str]]],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> list[tuple[str | None, str, int]]:
    """Pack each section's paragraphs into <= max_tokens chunks, no overlap.
    An oversized single paragraph becomes its own (oversized) chunk."""
    chunks: list[tuple[str | None, str, int]] = []
    for heading, paragraphs in sections:
        buf: list[str] = []
        buf_tokens = 0
        for para in paragraphs:
            n = len(tokenizer.encode(para, add_special_tokens=False))
            if buf and buf_tokens + n > max_tokens:
                chunks.append((heading, "\n\n".join(buf), buf_tokens))
                buf, buf_tokens = [], 0
            buf.append(para)
            buf_tokens += n
        if buf:
            chunks.append((heading, "\n\n".join(buf), buf_tokens))
    return chunks


def chunk_paper(
    markdown: str, tokenizer: PreTrainedTokenizerBase
) -> list[tuple[str | None, str, int]]:
    """Pure function: Markdown text -> [(section_heading, text, n_tokens), ...]."""
    sections = _split_sections(markdown)
    return _pack_chunks(sections, tokenizer, config.MAX_CHUNK_TOKENS)


def chunk(limit: int | None = None) -> None:
    con = duckdb.connect(str(config.DB_PATH))
    con.execute(_SCHEMA)

    already_chunked = {
        r[0] for r in con.execute("SELECT DISTINCT bibcode FROM chunks").fetchall()
    }
    all_papers = con.execute("SELECT bibcode, arxiv_id FROM papers").fetchall()
    todo = [
        (bibcode, arxiv_id)
        for bibcode, arxiv_id in all_papers
        if bibcode not in already_chunked
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
        rows = [
            (bibcode, arxiv_id, i, heading, text, n_tokens)
            for i, (heading, text, n_tokens) in enumerate(packed)
        ]
        con.executemany(_INSERT_CHUNK, rows)
        print(f"  [ok] {stem}: {len(rows)} chunks")

    con.commit()
    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk converted Markdown into passage-sized pieces")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending papers")
    args = parser.parse_args()

    chunk(limit=args.limit)

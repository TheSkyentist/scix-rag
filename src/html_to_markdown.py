"""
Convert arXiv HTML5 papers into clean Markdown for Arcana upload.

Rationale (measured on 2506.05459v1): raw arXiv HTML is ~75% markup
overhead, a quarter of which is MathML. Every <math> tag's `alttext`
attribute already holds the original LaTeX, so we swap MathML trees for
plain `$...$`/`$$...$$` instead of keeping the XML. Nav/header/footer and
the bibliography are dropped entirely -- the citation graph lives in
papers.duckdb, not in embedded text. Figures keep their captions only,
figure images are not embedded.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

_HEADING_RE = re.compile(r"^h[1-6]$")
_SECTION_CLASSES = {
    "ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_paragraph",
    "ltx_appendix", "ltx_subappendix",
}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _replace_math(soup: BeautifulSoup) -> None:
    """Swap every <math alttext="..."> for inline/display LaTeX before any text extraction."""
    for m in soup.find_all("math"):
        alttext = (m.get("alttext") or "").strip()
        if not alttext:
            continue
        delim = "$$" if m.get("display") == "block" else "$"
        m.replace_with(f" {delim}{alttext}{delim} ")


def _is_section(el: Tag) -> bool:
    return el.name == "section" and bool(_SECTION_CLASSES & set(el.get("class") or []))


def _is_renderable_leaf(el: Tag) -> bool:
    if _is_section(el):
        return True
    if el.name in ("figure", "table"):
        return True
    return el.name == "div" and "ltx_para" in (el.get("class") or [])


def _iter_content_blocks(el: Tag, skip_ids: set[int]) -> list[Tag]:
    """Depth-first collection of renderable content (sections, paragraphs,
    figures, tables), transparently descending into generic grouping
    elements (e.g. LaTeXML's `ltx_logical-block` wrapper div) that carry no
    semantic meaning of their own -- some documents put body content
    directly under one of these instead of under `<section>` tags."""
    blocks: list[Tag] = []
    for child in el.find_all(recursive=False):
        if not isinstance(child, Tag) or id(child) in skip_ids:
            continue
        if _is_section(child) and "ltx_bibliography" in (child.get("class") or []):
            continue
        if _is_renderable_leaf(child):
            blocks.append(child)
        else:
            blocks.extend(_iter_content_blocks(child, skip_ids))
    return blocks


def _render_table(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean_text(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _render_figure(fig: Tag) -> str:
    cap = fig.find("figcaption")
    cap_text = _clean_text(cap.get_text(" ")) if cap else ""
    table = fig.find("table")
    if table is not None:
        # LaTeXML wraps float tables as <figure class="ltx_table"><figcaption/><table/></figure>
        table_text = _render_table(table)
        parts = [p for p in (f"**Table.** {cap_text}" if cap_text else "", table_text) if p]
        return "\n".join(parts)
    return f"**Figure.** {cap_text}" if cap_text else ""


def _render_para_div(div: Tag) -> list[str]:
    paras = div.find_all("p", recursive=False)
    if not paras:
        text = _clean_text(div.get_text(" "))
        return [text] if text else []
    blocks = []
    for p in paras:
        text = _clean_text(p.get_text(" "))
        if text:
            blocks.append(text)
    return blocks


def _render_block(el: Tag | NavigableString, depth: int) -> list[str]:
    if isinstance(el, NavigableString):
        text = _clean_text(str(el))
        return [text] if text else []

    if _is_section(el):
        heading = el.find(lambda t: isinstance(t, Tag) and _HEADING_RE.match(t.name or ""))
        blocks = []
        if heading is not None:
            title = _clean_text(heading.get_text(" "))
            if title:
                blocks.append("#" * min(depth + 1, 6) + " " + title)
        for child in el.find_all(recursive=False):
            if child is heading:
                continue
            blocks.extend(_render_block(child, depth + 1))
        return blocks

    if el.name == "figure":
        rendered = _render_figure(el)
        return [rendered] if rendered else []

    if el.name == "table":
        rendered = _render_table(el)
        return [rendered] if rendered else []

    if el.name == "div" and "ltx_para" in (el.get("class") or []):
        return _render_para_div(el)

    # Generic fallback for anything else (acknowledgements, misc divs, ...):
    # flatten to one paragraph rather than silently dropping content.
    text = _clean_text(el.get_text(" "))
    return [text] if text else []


def convert(html: str) -> str:
    """Convert one arXiv HTML5 paper to clean Markdown."""
    soup = BeautifulSoup(html, "lxml")
    _replace_math(soup)

    article = soup.find("article", class_="ltx_document") or soup.find("body") or soup
    blocks: list[str] = []

    title_tag = article.find("h1", class_="ltx_title_document")
    if title_tag:
        blocks.append("# " + _clean_text(title_tag.get_text(" ")))

    abstract = article.find("div", class_="ltx_abstract")
    if abstract:
        paras = abstract.find_all("p")
        text = " ".join(_clean_text(p.get_text(" ")) for p in paras)
        if text:
            blocks.append("## Abstract")
            blocks.append(text)

    keywords = article.find("div", class_="ltx_keywords")
    if keywords:
        text = _clean_text(keywords.get_text(" ")).removeprefix("Key Words.").strip()
        if text:
            blocks.append(f"**Keywords:** {text}")

    authors = article.find("div", class_="ltx_authors")
    skip_ids = {id(t) for t in (title_tag, abstract, keywords, authors) if t is not None}

    for block_el in _iter_content_blocks(article, skip_ids):
        blocks.extend(_render_block(block_el, depth=1))

    return "\n\n".join(b for b in blocks if b)

"""
Convert whatever's sitting in data/html/ and data/pdf/ into Markdown,
deleting the raw file once conversion succeeds. Companion to fetch.py --
fetch grabs raw bytes, this does the actual conversion that gets kept.
Purely filesystem-driven: no database reads or writes.

HTML path: same cleaning fetch.py used to validate the source (MathML ->
LaTeX, bibliography dropped, nav/boilerplate stripped) -- see
html_to_markdown.py. Pure Python, no ML. Should essentially always
succeed, since fetch.py already validated it; if it doesn't, the file is
left in place rather than guessed at further -- worth a look, not a retry
loop.

PDF path: MinerU (`mineru-kit parse --tier standard --ocr-mode auto`).
Chosen over Docling+Tesseract and AnyDoc after a direct same-document
comparison: AnyDoc's local mode has no OCR at all (scanned pages need its
hosted cloud API -- disqualified outright, conflicts with this project's
local-only goal). Docling+Tesseract lost Greek letters/math in OCR
("Hα/Hβ" -> "Ha/HB") and flattened all headings to one level; MinerU's
standard tier preserved real LaTeX math, produced correct heading
*levels*, and was faster on OCR-heavy scanned pages. `--ocr-mode auto` is
used regardless of source: a nominally born-digital PDF can still have
individual image-only pages (observed directly in this corpus), and
per-page auto-detection handles that correctly where a document-level
split wouldn't.

Run directly (from the `extract` environment):
    pixi run convert
    pixi run convert --limit 5   # smoke-test on a handful of each type first
"""

from __future__ import annotations

import html
import re
import subprocess
import tempfile
from pathlib import Path

import config

from .html_to_markdown import convert as convert_html

_BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"^#{1,6}\s*(references|bibliography)\b", re.IGNORECASE | re.MULTILINE
)
# MinerU's standard tier embeds figure crops as inline base64 data URIs --
# we only want captions (same policy as html_to_markdown.py).
_BASE64_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)")
# ADS's own page-scan archive stamps this on every page; occasionally
# misfiled as a heading.
_WATERMARK_LINE_RE = re.compile(
    r"^#{0,6}.*Provided by the NASA Astrophysics Data System.*$\n?",
    re.IGNORECASE | re.MULTILINE,
)


def _convert_html_file(path: Path) -> str | None:
    md = convert_html(path.read_text(encoding="utf-8"))
    return md if len(md) >= config.MIN_MARKDOWN_CHARS else None


def _run_mineru(pdf_path: Path, out_md: Path) -> None:
    subprocess.run(
        [
            "mineru-kit", "parse", str(pdf_path),
            "-o", str(out_md),
            "--tier", "standard",
            "--ocr-mode", "auto",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _postprocess_pdf_md(md: str) -> str:
    md = html.unescape(md)
    md = _BASE64_IMAGE_RE.sub("", md)
    md = _WATERMARK_LINE_RE.sub("", md)
    match = _BIBLIOGRAPHY_HEADING_RE.search(md)
    if match:
        md = md[: match.start()]
    return md.strip()


def _convert_pdf_file(pdf_path: Path, tmp_dir: Path) -> str | None:
    tmp_md = tmp_dir / f"{pdf_path.stem}.md"
    try:
        _run_mineru(pdf_path, tmp_md)
        md = _postprocess_pdf_md(tmp_md.read_text(encoding="utf-8"))
    finally:
        tmp_md.unlink(missing_ok=True)
    return md if len(md) >= config.MIN_MARKDOWN_CHARS else None


def convert(limit: int | None) -> None:
    html_files = sorted(config.HTML_DIR.glob("*.html"))
    pdf_files = sorted(config.PDF_DIR.glob("*.pdf"))
    if limit:
        html_files = html_files[:limit]
        pdf_files = pdf_files[:limit]
    print(f"Converting {len(html_files)} HTML + {len(pdf_files)} PDF files")

    for path in html_files:
        md = _convert_html_file(path)
        if md:
            (config.MARKDOWN_DIR / f"{path.stem}.md").write_text(md, encoding="utf-8")
            path.unlink()
            print(f"  [ok]   {path.stem} ({len(md)} chars)")
        else:
            print(f"  [fail] {path.stem}: converted to too little, left in place for review")

    if not pdf_files:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        for path in pdf_files:
            try:
                md = _convert_pdf_file(path, Path(tmp_dir))
            except subprocess.CalledProcessError as e:
                print(f"  [fail] {path.stem}: mineru-kit exited {e.returncode}: {e.stderr[-500:]}")
                continue
            except Exception as e:
                print(f"  [fail] {path.stem}: {e}")
                continue

            if md:
                (config.MARKDOWN_DIR / f"{path.stem}.md").write_text(md, encoding="utf-8")
                path.unlink()
                print(f"  [ok]   {path.stem} ({len(md)} chars)")
            else:
                print(f"  [fail] {path.stem}: converted to too little, left in place for review")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert raw HTML/PDF into Markdown")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files per type")
    args = parser.parse_args()

    convert(limit=args.limit)

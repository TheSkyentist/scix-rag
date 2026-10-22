"""
Convert whatever's sitting in data/html/ and data/pdf/ into Markdown in
data/markdown/, deleting the raw file once conversion succeeds. Companion to
fetch.py: fetch grabs raw bytes, this does the conversion that gets kept.
Purely filesystem-driven: no database reads or writes.

HTML path: html_to_markdown.py (MathML -> LaTeX, bibliography dropped, nav and
boilerplate stripped). Pure Python, no ML. fetch.py already validated the
source with the same code, so it should essentially always succeed; if it
doesn't, the file is left in place for a look rather than retried.

PDF path: MinerU (`mineru-kit parse --tier standard --ocr-mode auto`), run
locally. `--ocr-mode auto` is used regardless of source because even a
nominally born-digital PDF can have individual image-only pages. This is by
the slowest step of the pipeline. The Markdown is then cleaned of the
references section, base64 figure crops and ADS page-scan watermarks.

Files named in `data/convert_ignore.txt` (one filename per line, `#` comments)
are skipped by the bulk run -- for PDFs that reliably time out or fail. The
bulk run appends to it itself (`name  # reason`) whenever a file times out
(MinerU's 1800 s limit), makes `mineru-kit` exit non-zero, or converts to too
little text. Other exceptions (I/O errors etc.) are only printed, so they retry
next run. Naming files explicitly on the command line converts them regardless
of the list and never records anything; `--retry-failed` does the same for the
whole bulk run.

Run directly (from the `convert` environment):
    pixi run convert
    pixi run convert --dry-run   # count what is waiting, convert nothing
    pixi run convert --limit 5   # smoke-test on a handful of each type first
    pixi run convert 1965ApJS....9..321M.pdf   # force specific file(s)
    pixi run convert --retry-failed            # also retry convert_ignore.txt entries
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
    r'^#{1,6}\s*(references|bibliography)\b', re.IGNORECASE | re.MULTILINE
)
# MinerU's standard tier embeds figure crops as inline base64 data URIs --
# we only want captions (same policy as html_to_markdown.py).
_BASE64_IMAGE_RE = re.compile(r'!\[[^\]]*\]\(data:image/[^)]+\)')
# ADS's own page-scan archive stamps this on every page; occasionally
# misfiled as a heading.
_WATERMARK_LINE_RE = re.compile(
    r'^#{0,6}.*Provided by the NASA Astrophysics Data System.*$\n?',
    re.IGNORECASE | re.MULTILINE,
)


def _convert_html_file(path: Path) -> str | None:
    md = convert_html(path.read_text(encoding='utf-8'))
    return md if len(md) >= config.MIN_MARKDOWN_CHARS else None


def _run_mineru(pdf_path: Path, out_md: Path) -> None:
    subprocess.run(
        [
            'mineru-kit',
            'parse',
            str(pdf_path),
            '-o',
            str(out_md),
            '--tier',
            'standard',
            '--ocr-mode',
            'auto',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _postprocess_pdf_md(md: str) -> str:
    md = html.unescape(md)
    md = _BASE64_IMAGE_RE.sub('', md)
    md = _WATERMARK_LINE_RE.sub('', md)
    match = _BIBLIOGRAPHY_HEADING_RE.search(md)
    if match:
        md = md[: match.start()]
    return md.strip()


def _convert_pdf_file(pdf_path: Path, tmp_dir: Path) -> str | None:
    tmp_md = tmp_dir / f'{pdf_path.stem}.md'
    try:
        _run_mineru(pdf_path, tmp_md)
        md = _postprocess_pdf_md(tmp_md.read_text(encoding='utf-8'))
    finally:
        tmp_md.unlink(missing_ok=True)
    return md if len(md) >= config.MIN_MARKDOWN_CHARS else None


def _load_ignored() -> set[str]:
    if not config.CONVERT_IGNORE_PATH.exists():
        return set()
    lines = config.CONVERT_IGNORE_PATH.read_text(encoding='utf-8').splitlines()
    return {name for line in lines if (name := line.split('#', 1)[0].strip())}


def _record_failed(name: str, reason: str, ignored: set[str]) -> None:
    if name in ignored:
        return
    path = config.CONVERT_IGNORE_PATH
    needs_newline = (
        path.exists()
        and path.stat().st_size
        and not path.read_text(encoding='utf-8').endswith('\n')
    )
    with path.open('a', encoding='utf-8') as f:
        if needs_newline:
            f.write('\n')
        f.write(f'{name}  # {reason}\n')
    ignored.add(name)


def _resolve_targets(targets: list[str]) -> tuple[list[Path], list[Path]]:
    html_files: list[Path] = []
    pdf_files: list[Path] = []
    for target in targets:
        name = Path(target).name
        if name.endswith('.html') and (config.HTML_DIR / name).exists():
            html_files.append(config.HTML_DIR / name)
        elif name.endswith('.pdf') and (config.PDF_DIR / name).exists():
            pdf_files.append(config.PDF_DIR / name)
        else:
            print(f'  [skip] {target}: not found in data/html or data/pdf')
    return html_files, pdf_files


def convert(
    limit: int | None,
    targets: list[str] | None = None,
    retry_failed: bool = False,
    dry_run: bool = False,
) -> None:
    record = not targets  # explicit targets are deliberate retries; never recorded
    ignored: set[str] = set()
    if targets:
        html_files, pdf_files = _resolve_targets(targets)
        ignored_note = ''
    else:
        ignored = _load_ignored()
        html_files = sorted(config.HTML_DIR.glob('*.html'))
        pdf_files = sorted(config.PDF_DIR.glob('*.pdf'))
        n_total = len(html_files) + len(pdf_files)
        # Filter before --limit so ignored files don't eat the limit.
        if not retry_failed:
            html_files = [p for p in html_files if p.name not in ignored]
            pdf_files = [p for p in pdf_files if p.name not in ignored]
        n_ignored = n_total - len(html_files) - len(pdf_files)
        ignored_note = f' ({n_ignored} ignored)' if n_ignored else ''
        if limit:
            html_files = html_files[:limit]
            pdf_files = pdf_files[:limit]
    if dry_run:
        print('Dry run, nothing converted.')
        print(
            f'  {len(html_files)} HTML + {len(pdf_files)} PDF files would be converted{ignored_note}'
        )
        if pdf_files:
            print(
                '  PDFs are the slowest step: MinerU can take up to 30 minutes per paper'
            )
        return
    print(
        f'Converting {len(html_files)} HTML + {len(pdf_files)} PDF files{ignored_note}'
    )

    n_ok = n_failed = 0
    for path in html_files:
        md = _convert_html_file(path)
        if md:
            (config.MARKDOWN_DIR / f'{path.stem}.md').write_text(md, encoding='utf-8')
            path.unlink()
            n_ok += 1
            print(f'  [ok]   {path.stem} ({len(md)} chars)')
        else:
            n_failed += 1
            print(
                f'  [fail] {path.stem}: converted to too little, left in place for review'
            )
            if record:
                _record_failed(path.name, 'converted to too little text', ignored)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for path in pdf_files:
            try:
                md = _convert_pdf_file(path, Path(tmp_dir))
            except subprocess.TimeoutExpired:
                n_failed += 1
                print(f'  [fail] {path.stem}: mineru-kit timed out')
                if record:
                    _record_failed(path.name, 'MinerU timeout (1800s)', ignored)
                continue
            except subprocess.CalledProcessError as e:
                n_failed += 1
                print(
                    f'  [fail] {path.stem}: mineru-kit exited {e.returncode}: {e.stderr[-500:]}'
                )
                if record:
                    _record_failed(
                        path.name, f'mineru-kit exited {e.returncode}', ignored
                    )
                continue
            except Exception as e:
                n_failed += 1
                print(f'  [fail] {path.stem}: {e}')
                continue

            if md:
                (config.MARKDOWN_DIR / f'{path.stem}.md').write_text(
                    md, encoding='utf-8'
                )
                path.unlink()
                n_ok += 1
                print(f'  [ok]   {path.stem} ({len(md)} chars)')
            else:
                n_failed += 1
                print(
                    f'  [fail] {path.stem}: converted to too little, left in place for review'
                )
                if record:
                    _record_failed(path.name, 'converted to too little text', ignored)

    print(f'\nDone. {n_ok} converted, {n_failed} failed')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert raw HTML/PDF into Markdown')
    parser.add_argument(
        'targets',
        nargs='*',
        help='Specific filenames in data/html or data/pdf to convert, bypassing convert_ignore.txt',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only process the first N files per type',
    )
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='Also retry files listed in convert_ignore.txt',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Count the files that are waiting, convert nothing',
    )
    args = parser.parse_args()

    convert(
        limit=args.limit,
        targets=args.targets,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
    )

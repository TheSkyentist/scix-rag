import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
HTML_DIR = DATA_DIR / 'html'
MARKDOWN_DIR = DATA_DIR / 'markdown'
PDF_DIR = DATA_DIR / 'pdf'

# Hand-maintained list of raw filenames the bulk `convert` skips. Lives under
# data/ (gitignored); back it up along with papers.duckdb.
CONVERT_IGNORE_PATH = DATA_DIR / 'convert_ignore.txt'

# Papers `fetch` found nothing fetchable for (appended automatically; one
# stem per line with a `# reason`). Skipped on later runs unless --retry-failed.
FETCH_IGNORE_PATH = DATA_DIR / 'fetch_ignore.txt'

DATA_DIR.mkdir(exist_ok=True)
HTML_DIR.mkdir(exist_ok=True)
MARKDOWN_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# Optional at import so non-ADS stages (search, chunk, embed) run without .env;
# src/ads_query.py raises a clear error when it actually needs the key.
ADS_API_KEY = os.environ.get('ADS_API_KEY')
ADS_BASE_URL = 'https://api.adsabs.harvard.edu/v1'
ADS_FIELDS = (
    'bibcode,alternate_bibcode,identifier,title,author,first_author,'
    'abstract,year,pubdate,doctype,pub,volume,page,doi,keyword,'
    'arxiv_class,citation_count,reference,citation,property,esources'
)

MAX_CHUNK_TOKENS = 1024
MAX_TABLE_EMBED_TOKENS = 256  # tables are stored whole but only caption + header row get embedded

# Embedding stage (src/embed.py)
# Embedding models: id (stored as chunk_embeddings.model) -> (HF repo, text between "Query:" and the
# query). Any Qwen3-backbone, last-token-pooled embedder loads the same way (src/embed.py), e.g.
# 'Qwen3-Embedding-0.6B': ('Qwen/Qwen3-Embedding-0.6B', ''); bf16 only (8-/4-bit were slower, not
# faster). EMBED_MODEL is what `embed` writes, `search` reads and `chunk` takes its tokenizer from.
EMBED_MODELS = {
    'harrier-oss-v1-0.6b': ('microsoft/harrier-oss-v1-0.6b', ' '),
}
EMBED_MODEL = 'harrier-oss-v1-0.6b'  # chosen by a retrieval eval in Oct 2026; see CLAUDE.md
MAX_EMBED_TOKENS = 8192  # chunks longer than this are truncated (model context is 32K)
EMBED_BATCH_TOKENS = 16384  # max padded tokens (items x longest) per forward pass
EMBED_MAX_BATCH = 16
EMBED_COMMIT_TOKENS = 250_000  # ~tokens embedded between DB commits; bounds what Ctrl-C can lose
EMBED_TITLE_PREFIX = True  # prepend the paper title to each chunk's embedded text (open question)
# Queries (not documents) are formatted "Instruct: {task}\nQuery:{sep}{query}" (sep per EMBED_MODELS).
QUERY_INSTRUCTION = (
    'Given a question or topic from astrophysics research, retrieve passages '
    'from scientific papers that answer or discuss it'
)
ADS_PAGE_SIZE = 200
ADS_BATCH_SIZE = 75  # bibcodes per OR'd batch-fetch query
ADS_DELAY = 0.5  # seconds between ADS requests
ARXIV_DELAY = 1.0  # seconds between arXiv HTML fetches

# Below this, treat a conversion (HTML or PDF path) as empty/failed.
# Calibrated against the corpus: smallest genuine converted paper so far
# is ~1900 chars; ar5iv's own "conversion had a fatal error" placeholder
# still contains an (empty) <article class="ltx_document"> plus a stray
# error <div>, so a naive "ltx_document in html" check isn't enough --
# validate by actually running the real converter and checking it
# produced substance.
MIN_MARKDOWN_CHARS = 500

DB_PATH = DATA_DIR / 'papers.duckdb'

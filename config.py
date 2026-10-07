import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
HTML_DIR = DATA_DIR / 'html'
MARKDOWN_DIR = DATA_DIR / 'markdown'
PDF_DIR = DATA_DIR / 'pdf'

# Hand-maintained list of raw filenames the bulk `convert` skips. Lives at
# the repo root (not under data/) so it's committed, unlike data/pdf/.
CONVERT_IGNORE_PATH = ROOT / 'convert_ignore.txt'

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
EMBED_MODEL_SIZE = '0.6B'  # Qwen3-Embedding-{0.6B,4B}; bf16 only (8-/4-bit were slower, not faster)
MAX_EMBED_TOKENS = 8192  # chunks longer than this are truncated (model context is 32K)
EMBED_BATCH_TOKENS = 16384  # max padded tokens (items x longest) per forward pass
EMBED_MAX_BATCH = 16
EMBED_COMMIT_TOKENS = 250_000  # ~tokens embedded between DB commits; bounds what Ctrl-C can lose
EMBED_TITLE_PREFIX = True  # prepend the paper title to each chunk's embedded text (open question)
# Qwen3-Embedding wants queries (not documents) as "Instruct: {task}\nQuery:{query}".
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

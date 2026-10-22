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

# Created on import so every stage can write to them
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

# Embedding stage (src/embed.py)
# EMBED_MODEL must be a Qwen3-backbone embedding model (a Hugging Face repo id).
# QUERY_INSTRUCTION must format queries the way your model expects; check its model card.
EMBED_MODEL = 'microsoft/harrier-oss-v1-0.6b'
QUERY_INSTRUCTION = (
    'Instruct: Given a question or topic from astronomy or astrophysics research, '
    'retrieve passages from scientific papers that answer or discuss it\nQuery: {query}'
)

# Chunking stage (src/chunk.py)
MAX_CHUNK_TOKENS = 1024  # target tokens per chunk; one over-long sentence can exceed it
# Tables are stored whole, but only the caption and header row (up to this) are embedded
MAX_TABLE_EMBED_TOKENS = 256

# Embedding stage: tuning (src/embed.py)
MAX_EMBED_TOKENS = 8192  # longer chunks are truncated
EMBED_BATCH_TOKENS = 16384  # max padded tokens per forward pass
EMBED_MAX_BATCH = 16  # max chunks per forward pass
EMBED_COMMIT_TOKENS = (
    250_000  # tokens embedded per DB commit; Ctrl-C loses at most this
)
# Prepend the paper title to each chunk's embedded text; changing it re-embeds the chunks
EMBED_TITLE_PREFIX = True

# ADS and arXiv requests (src/ads_query.py, src/fetch.py)
ADS_PAGE_SIZE = 200  # seed records per ADS search request
ADS_BATCH_SIZE = 75  # bibcodes per OR'd batch-fetch query
ADS_DELAY = 0.5  # seconds between ADS requests
ARXIV_DELAY = 1.0  # seconds between arXiv HTML fetches

# Below this many characters a conversion counts as empty/failed (the smallest genuine
# paper so far is ~1,900). Validated by running the real converter, since ar5iv's error
# placeholder still contains an empty <article class="ltx_document">.
MIN_MARKDOWN_CHARS = 500

DB_PATH = DATA_DIR / 'papers.duckdb'

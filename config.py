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

ADS_API_KEY = os.environ['ADS_API_KEY']
ADS_BASE_URL = 'https://api.adsabs.harvard.edu/v1'
ADS_FIELDS = (
    'bibcode,alternate_bibcode,identifier,title,author,first_author,'
    'abstract,year,pubdate,doctype,pub,volume,page,doi,keyword,'
    'arxiv_class,citation_count,reference,citation,property,esources'
)

MAX_CHUNK_TOKENS = 1024
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

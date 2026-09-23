import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
HTML_DIR = DATA_DIR / 'html'
MARKDOWN_DIR = DATA_DIR / 'markdown'
PDF_DIR = DATA_DIR / 'pdf'

DATA_DIR.mkdir(exist_ok=True)
HTML_DIR.mkdir(exist_ok=True)
MARKDOWN_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

ADS_API_KEY = os.environ['ADS_API_KEY']
ADS_BASE_URL = 'https://api.adsabs.harvard.edu/v1'
ADS_FIELDS = (
    'bibcode,alternate_bibcode,identifier,title,author,first_author,'
    'abstract,year,pubdate,doctype,pub,volume,page,doi,keyword,'
    'arxiv_class,citation_count,reference,citation,property'
)

# Not needed until we start uploading to Arcana; left optional so config
# stays importable with only ADS_API_KEY set. Env var name(s) TBD.
ARCANA_TOKEN = os.environ.get('SAIA_API_KEY')
ARCANA_NAME = os.environ.get('ARCANA_NAME')
ARCANA_BASE_URL = 'https://chat-ai.academiccloud.de/v1/arcanas/api/v1'
ARCANA_MAX_FILE_BYTES = 300 * 1024 * 1024  # documented per-file hard cap

ARCANA_INDEX_BATCH_SIZE = 500

MAX_CHUNK_TOKENS = 1024
ADS_PAGE_SIZE = 200
ADS_BATCH_SIZE = 75  # bibcodes per OR'd batch-fetch query
ADS_DELAY = 0.5  # seconds between ADS requests
ARXIV_DELAY = 3.0  # seconds between arXiv HTML fetches

PAPERS_PATH = (
    DATA_DIR / 'papers.parquet'
)  # legacy, used by fetch_html/chunk/arcana_ingest for now
CHUNKS_PATH = DATA_DIR / 'chunks.parquet'  # legacy, ditto
DB_PATH = DATA_DIR / 'papers.duckdb'

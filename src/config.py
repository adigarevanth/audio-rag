"""Configuration and settings loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# --- Paths ---
PROJECT_ROOT = _project_root
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
EVAL_DIR = DATA_DIR / "eval"
METADATA_PATH = DATA_DIR / "metadata.json"

# Ensure directories exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# --- HuggingFace ---
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Postgres ---
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "audio_rag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "audio_rag")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "audio_rag_dev")

DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# --- Models ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# --- Chunking ---
MAX_CHUNK_WORDS = 375  # ~500 tokens; split long turns at sentence boundaries
MIN_CHUNK_WORDS = 10   # merge very short turns from the same speaker

# --- Search ---
RRF_K = 60             # RRF damping constant
SEARCH_K = 20          # number of results per individual search strategy

# Audio RAG: Hybrid Search Over Audio Transcripts

Search across audio recordings by keyword and meaning. Given a collection of two-speaker conversations, this system transcribes, diarizes (identifies who spoke when), embeds, and indexes them into Postgres with pgvector. Search results return the matching file, speaker, timestamp, and text.

## Architecture

```
Audio Files → faster-whisper (transcription) → pyannote (diarization) → alignment
→ speaker-turn chunks → embeddings (all-MiniLM-L6-v2) + tsvector → Postgres + pgvector
→ hybrid search (keyword + semantic + RRF fusion)
```

---

## Prerequisites

- **WSL 2** with Ubuntu 22.04+ (or any Debian-based distro)
- **Docker Desktop** with WSL 2 backend enabled
- **Python 3.11+**
- **uv** (Python package manager)
- **ffmpeg** (for audio processing)
- **A free HuggingFace account** (for pyannote model access)

---

## Setup in WSL

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg git curl
```

If `python3.11` is not available in your distro's repos:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell or run `source $HOME/.local/bin/env` so the `uv` command is on your PATH.

### 3. Clone the repo

```bash
git clone <your-repo-url> audio-rag
cd audio-rag
```

### 4. Create the virtual environment and install dependencies

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

This installs everything: faster-whisper, pyannote.audio, sentence-transformers, psycopg, pgvector, yt-dlp, pytest, etc.

> **Note:** The first install may take a few minutes as it downloads PyTorch and the ML libraries.

### 5. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your HuggingFace token:

```bash
nano .env
```

```
HF_TOKEN=hf_your_actual_token_here
```

**To get the token:**
1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the terms
3. Go to https://huggingface.co/pyannote/segmentation-3.0 and accept the terms
4. Create a token at https://huggingface.co/settings/tokens (read access is enough)

### 6. Start Postgres with pgvector

Make sure Docker Desktop is running with WSL 2 integration enabled.

```bash
docker compose up -d
```

Verify it's running:

```bash
docker compose ps
```

You should see `audio-rag-db` with status `Up`. The database schema is created automatically on first start via `scripts/init_db.sql`.

To check the database is accessible:

```bash
docker exec -it audio-rag-db psql -U audio_rag -d audio_rag -c "SELECT 1;"
```

---

## Prepare the Golden Dataset

### 1. Source audio files

You need 5-6 audio files, each 8-10 minutes long, each with exactly 2 speakers. Good free sources include podcast episodes and interviews on YouTube.

**Download from YouTube using yt-dlp** (installed as a project dependency):

```bash
# Download audio only
yt-dlp -x --audio-format wav -o "data/audio/episode_01.%(ext)s" "https://youtube.com/watch?v=VIDEO_ID"
```

### 2. Trim and convert audio

Trim to a 10-minute segment starting at a given timestamp:

```bash
# Trim to 10 minutes starting at 5:00
ffmpeg -i data/audio/episode_01.wav -ss 00:05:00 -t 00:10:00 -ar 16000 -ac 1 data/audio/episode_01_trimmed.wav

# Rename
mv data/audio/episode_01_trimmed.wav data/audio/episode_01.wav
```

Batch convert all files to 16kHz mono WAV (if not already):

```bash
for f in data/audio/*.wav; do
  ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.wav}_16k.wav" && mv "${f%.wav}_16k.wav" "$f"
done
```

### 3. Update metadata.json

Edit `data/metadata.json` with the real file names, sources, and speaker names:

```json
{
  "files": [
    {
      "filename": "episode_01.wav",
      "source_url": "https://youtube.com/watch?v=...",
      "topic": "AI in healthcare",
      "duration_s": 600,
      "num_speakers": 2,
      "speaker_map": {
        "SPEAKER_00": "Dr. Jane Smith",
        "SPEAKER_01": "Host John"
      }
    }
  ]
}
```

> **Tip:** You can leave the `speaker_map` as generic names first, run the pipeline, inspect the transcripts in `data/transcripts/`, figure out which `SPEAKER_00`/`SPEAKER_01` is who, then update the map and re-index with `--force`.

---

## Running the Pipeline

### Index all audio files

```bash
audio-rag index
```

This runs the full pipeline for every file in `metadata.json`:
1. Transcribes with faster-whisper (downloads the model on first run)
2. Diarizes with pyannote (downloads the model on first run)
3. Aligns words to speakers
4. Chunks by speaker turn
5. Embeds with all-MiniLM-L6-v2 (downloads on first run)
6. Stores chunks + embeddings in Postgres

**First run will be slow** — model downloads + CPU-based transcription. Subsequent runs use cached transcripts and diarization results from `data/transcripts/`.

To re-index a specific file:

```bash
audio-rag index --file episode_01.wav --force
```

### Search

```bash
# Hybrid search (keyword + semantic, default)
audio-rag search "machine learning in medicine"

# Keyword only
audio-rag search "neural network" --mode keyword

# Semantic only
audio-rag search "artificial intelligence applications" --mode semantic

# Limit results
audio-rag search "climate change" -k 5
```

Output shows each result's file, speaker, timestamp, match type, and text preview:

```
================================================================================
  Search results for: "machine learning in medicine"  (mode: hybrid, top 10)
================================================================================

  [1] episode_03.wav  |  Dr. Jane Smith  |  02:04 - 02:28  |  both
      Score: 0.0323
      "We've been applying deep learning models to radiology images and the results..."

  [2] episode_01.wav  |  Host John  |  05:12 - 05:30  |  semantic
      Score: 0.0164
      "So you're saying these algorithms can actually diagnose conditions that..."
```

### Run evaluation

First, fill in `data/eval/golden_queries.json` with labeled queries and expected results (do this after inspecting the transcripts). Then:

```bash
# Via CLI
audio-rag evaluate

# Or via pytest
uv run pytest tests/test_search.py -v
```

### Run unit tests

```bash
uv run pytest tests/test_pipeline.py -v
```

---

## Commands Reference

| Command | Description |
|---|---|
| `audio-rag index` | Index all audio files from metadata.json |
| `audio-rag index --file X.wav` | Index a single file |
| `audio-rag index --force` | Re-index everything from scratch |
| `audio-rag search "query"` | Hybrid search (default) |
| `audio-rag search "query" --mode keyword` | Keyword-only search |
| `audio-rag search "query" --mode semantic` | Semantic-only search |
| `audio-rag search "query" -k 5` | Return top 5 results |
| `audio-rag evaluate` | Run recall@k evaluation |
| `audio-rag -v index` | Verbose/debug logging |

---

## Troubleshooting

**"HF_TOKEN is not set"**
Make sure you copied `.env.example` to `.env` and added your HuggingFace token. Also ensure you accepted the pyannote model terms on HuggingFace.

**"connection refused" on Postgres**
Check Docker is running: `docker compose ps`. If the container is not up, run `docker compose up -d`. If you're running WSL and Docker Desktop, make sure WSL integration is enabled in Docker Desktop settings.

**Transcription is very slow**
faster-whisper runs on CPU by default. The `base` model takes roughly 2-3x real-time on a modern CPU (so 10 min audio takes ~20-30 min). Use `WHISPER_MODEL=tiny` in `.env` for faster but less accurate transcription. If you have an NVIDIA GPU accessible in WSL, install the CUDA version of CTranslate2 for significant speedup.

**"No module named src"**
Make sure you installed the project in editable mode: `uv pip install -e .` from the repo root with the venv activated.

**uv can't find Python 3.11**
Install it through uv itself: `uv python install 3.11`, then re-run `uv venv --python 3.11`.

**Docker volume issues**
If the database seems empty after restarting Docker, check that the volume persists: `docker volume ls` should show an `audio-rag_pgdata` volume. If you need to start fresh: `docker compose down -v && docker compose up -d`.

---

## Project Structure

```
audio-rag/
├── .env.example              # Environment variables template
├── .gitignore
├── docker-compose.yml        # Postgres + pgvector
├── pyproject.toml            # Python dependencies (uv)
├── scripts/
│   └── init_db.sql           # Database schema
├── data/
│   ├── audio/                # Audio files (WAV, 16kHz, mono)
│   ├── transcripts/          # Generated transcripts + diarization (cached)
│   ├── metadata.json         # Audio file metadata and speaker maps
│   └── eval/
│       └── golden_queries.json  # Labeled queries for evaluation
├── src/
│   ├── config.py             # Settings from .env
│   ├── transcribe.py         # faster-whisper transcription
│   ├── diarize.py            # pyannote speaker diarization
│   ├── align.py              # Merge transcription + diarization
│   ├── chunk.py              # Speaker-turn chunking
│   ├── embed.py              # Sentence-transformer embeddings
│   ├── db.py                 # Postgres/pgvector operations
│   ├── index.py              # Full indexing pipeline
│   ├── search.py             # Hybrid search (keyword + semantic + RRF)
│   └── cli.py                # CLI entry point
└── tests/
    ├── test_pipeline.py      # Unit tests for alignment and chunking
    └── test_search.py        # Recall@k evaluation tests
```

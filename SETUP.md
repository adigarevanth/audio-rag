# Reviewer Setup Guide

This guide walks you through setting up, running, and evaluating the audio-rag hybrid search system from scratch. The repo includes pre-built audio files, transcripts, diarization output, and a golden evaluation dataset so you can go from clone to results quickly.

**Estimated time:** ~15 minutes for setup + indexing (transcription and diarization are pre-cached)

---

## Prerequisites

- **Docker** (for Postgres + pgvector)
- **Python 3.11+**
- **uv** (Python package manager)
- **ffmpeg**

On Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg git curl
```

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

---

## Step 1: Clone and Install

```bash
git clone <repo-url> audio-rag
cd audio-rag
```

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add a HuggingFace token:

```
HF_TOKEN=hf_your_token_here
```

**Getting the token (free):**
1. Create an account at https://huggingface.co
2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept model terms at https://huggingface.co/pyannote/segmentation-3.0
4. Create token at https://huggingface.co/settings/tokens

> The HF token is needed for speaker diarization. If you only want to test search and evaluation (skipping the diarization step), the pre-cached transcripts and diarization files in `data/transcripts/` mean you won't need to re-run diarization.

---

## Step 3: Start the Database

```bash
docker compose up -d
```

Verify it's running:

```bash
docker compose ps
```

You should see `audio-rag-db` with status `Up` and port `5432` mapped. The schema is created automatically on first start.

Verify database connectivity:

```bash
docker exec -it audio-rag-db psql -U audio_rag -d audio_rag -c "\dt"
```

You should see the `audio_files` and `chunks` tables.

---

## Step 4: Index the Audio Files

The repo includes 5 pre-processed audio files with cached transcripts and diarization in `data/transcripts/`. Indexing will use these cached files, so it only needs to run alignment, chunking, embedding, and database insertion. No GPU or long transcription wait required.

```bash
audio-rag index
```

You should see output like:

```
HH:MM:SS [INFO] src.index: Using cached transcript for 'andrew_ng_ai_opportunities.wav'
HH:MM:SS [INFO] src.index: Using cached diarization for 'andrew_ng_ai_opportunities.wav'
HH:MM:SS [INFO] src.align: Aligned 1870 words to speakers ['SPEAKER_00', 'SPEAKER_01'] (0 unassigned)
HH:MM:SS [INFO] src.chunk: Created 42 chunks from 38 raw turns
HH:MM:SS [INFO] src.embed: Embedding 42 chunks...
HH:MM:SS [INFO] src.db: Inserted 42 chunks for audio_file_id=1
HH:MM:SS [INFO] src.index: Successfully indexed 'andrew_ng_ai_opportunities.wav' (42 chunks)
...
```

This takes ~2-5 minutes (embedding model downloads on first run, then embeds all chunks).

To verify data is in the database:

```bash
docker exec -it audio-rag-db psql -U audio_rag -d audio_rag -c "SELECT COUNT(*) FROM chunks;"
docker exec -it audio-rag-db psql -U audio_rag -d audio_rag -c "SELECT filename, speaker_a, speaker_b FROM audio_files;"
```

---

## Step 5: Try Searching

Run a few searches to see the system in action:

```bash
# Hybrid search (keyword + semantic combined via RRF)
audio-rag search "Augustus Caesar imperial monarchy"

# Semantic search (finds results even with different wording)
audio-rag search "are we living in a computer simulation"

# Keyword search (exact term matching)
audio-rag search "Sam Altman accused of murder" --mode keyword

# Compare: this semantic query returns nothing in keyword mode
audio-rag search "what caused the fall of Constantinople" --mode keyword
audio-rag search "what caused the fall of Constantinople" --mode semantic
```

Each result shows:
- The audio file it came from
- Which speaker said it
- Timestamp range (MM:SS)
- Match type (keyword, semantic, or both)
- Text preview

---

## Step 6: Run the Evaluation

The golden dataset at `data/eval/golden_queries.json` contains 36 labeled queries across three categories (keyword, semantic, hybrid) with 43 total expected results.

### Option A: CLI evaluation (prints results + saves CSV)

```bash
audio-rag evaluate
```

This runs all 36 queries against three search strategies (hybrid, keyword-only, semantic-only), computes Recall@k, Precision@k, and MRR, and saves a detailed CSV to `data/eval/`.

Expected output:

```
Evaluating against 36 golden queries...

--- Hybrid Search ---
  recall@5: 0.929
  recall@10: 0.957
  precision@5: 0.349
  precision@10: 0.234
  mrr: 0.858

--- Keyword Only ---
  recall@5: 0.386
  recall@10: 0.386
  ...

--- Semantic Only ---
  recall@5: 0.929
  recall@10: 0.957
  ...

--- By Query Category ---
  Hybrid:
    hybrid     ( 8 queries)  recall@10=1.000  mrr=1.000
    keyword    (12 queries)  recall@10=0.917  mrr=0.767
    semantic   (15 queries)  recall@10=0.967  mrr=0.856

📄 Results saved to: data/eval/eval_results_YYYYMMDD_HHMMSS.csv
```

### Option B: pytest (for pass/fail assertions)

```bash
uv run pytest tests/test_search.py -v
```

This runs four tests:
- `test_recall_at_5` — asserts Recall@5 >= 0.70
- `test_recall_at_10` — asserts Recall@10 >= 0.85
- `test_mrr` — asserts MRR >= 0.60
- `test_hybrid_beats_individual` — asserts hybrid Recall@10 >= best individual strategy

### Option C: Unit tests (no database needed)

```bash
uv run pytest tests/test_pipeline.py -v
```

Tests alignment and chunking logic with synthetic data. No database or audio files required.

---

## Step 7: Review the Results

After running the evaluation:

- **CSV file** in `data/eval/` contains per-query results for all three strategies, aggregate metrics, and per-category breakdowns
- **EVAL_OBSERVATIONS.md** contains our analysis of the results, metric definitions, key observations, and limitations
- **IMPLEMENTATION_PLAN.md** documents the architectural decisions and rationale

---

## What's Included in the Repo

| Path | Description |
|---|---|
| `data/audio/*.wav` | 5 golden dataset audio files (8-10 min each, 2 speakers) |
| `data/transcripts/*_transcript.json` | Cached word-level transcripts from faster-whisper |
| `data/transcripts/*_diarization.json` | Cached speaker diarization from pyannote |
| `data/metadata.json` | Audio file metadata with speaker name mappings |
| `data/eval/golden_queries.json` | 36 labeled queries with expected results |
| `data/eval/eval_results_*.csv` | Evaluation results (generated by `audio-rag evaluate`) |
| `src/` | All source code (pipeline + search + CLI) |
| `tests/` | Unit tests + evaluation tests |
| `IMPLEMENTATION_PLAN.md` | Architecture and design decisions |
| `EVAL_OBSERVATIONS.md` | Evaluation methodology, metrics, and findings |

---

## Quick Reference

| Command | What It Does |
|---|---|
| `docker compose up -d` | Start Postgres |
| `audio-rag index` | Index all audio files |
| `audio-rag search "query"` | Hybrid search |
| `audio-rag search "query" --mode keyword` | Keyword-only search |
| `audio-rag search "query" --mode semantic` | Semantic-only search |
| `audio-rag evaluate` | Run full evaluation + save CSV |
| `uv run pytest tests/ -v` | Run all tests |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `connection refused` on Postgres | Run `docker compose up -d` and check with `docker compose ps` |
| `HF_TOKEN is not set` | Copy `.env.example` to `.env` and add your HuggingFace token |
| `No module named src` | Run `uv pip install -e .` from the repo root |
| Slow indexing | Cached transcripts skip transcription/diarization. If re-running from scratch, expect ~15 min per file on CPU |
| `docker compose ps` shows no ports | Run `docker compose down && docker compose up -d` to recreate the container |

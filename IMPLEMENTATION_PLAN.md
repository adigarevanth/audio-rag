# Audio RAG: Hybrid Search Over Audio Transcripts

## Implementation Plan

---

## 1. Architecture Overview

```
Audio Files (5-6 × 8-10 min, 2 speakers each)
        │
        ▼
┌───────────────────┐
│  Transcription     │  ← faster-whisper (local, free)
│  (word-level ts)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Diarization       │  ← pyannote.audio 3.x (local)
│  (speaker labels)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Alignment &       │  Merge Whisper word timestamps with
│  Chunking          │  pyannote speaker segments → speaker turns
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Indexing           │
│  ├─ Embeddings      │  ← all-MiniLM-L6-v2 (local, 384-dim)
│  └─ Full-text       │  ← Postgres tsvector/tsquery
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Postgres + pgvector│  ← Docker (pgvector/pgvector:pg16)
│  (hybrid storage)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Hybrid Search      │  ← RRF over keyword + semantic results
│  API / CLI          │
└───────────────────┘
```

---

## 2. Tech Stack

| Component         | Choice                          | Rationale                                                  |
|-------------------|---------------------------------|------------------------------------------------------------|
| Language          | Python 3.11+                    | Best ecosystem for audio/ML pipelines                      |
| Transcription     | faster-whisper (local)          | Free, fast on CPU (CTranslate2), word-level timestamps     |
| Diarization       | pyannote.audio 3.x              | SOTA open-source; free with HuggingFace token              |
| Embeddings        | all-MiniLM-L6-v2                | Fast (384-dim), strong quality, runs on CPU                |
| Database          | Postgres 16 + pgvector          | Hybrid storage: vectors + full-text in one DB              |
| Container         | Docker (pgvector/pgvector:pg16) | Reproducible, zero-install database setup                  |
| Hybrid search     | RRF in application code         | Simple, effective rank fusion; easy to tune                |
| Chunking          | By speaker turn                 | Natural boundary, preserves speaker attribution            |

---

## 3. Golden Dataset

### Suggested Sources

We need 5-6 audio files, each 8-10 minutes, each with exactly 2 speakers. Good free sources:

| # | Source Idea                            | Where to Find                                              |
|---|----------------------------------------|------------------------------------------------------------|
| 1 | NPR Short Wave podcast episode         | npr.org — science interviews, clear 2-speaker format       |
| 2 | Lex Fridman clip (trimmed to 10 min)   | YouTube — long-form interviews, easily trimmable           |
| 3 | BBC HARDtalk interview segment         | YouTube/BBC Sounds — distinct interviewer + guest           |
| 4 | TED Interview podcast episode          | ted.com — structured host + guest conversations            |
| 5 | The Vergecast interview segment        | YouTube — tech-focused, clear 2-speaker segments           |
| 6 | Huberman Lab guest clip (10 min)       | YouTube — science/health, distinct host + expert            |

### Dataset Preparation

1. Download audio using `yt-dlp` (for YouTube sources) or direct download
2. Trim to 8-10 minute segments using `ffmpeg`
3. Convert to consistent format: WAV, 16kHz, mono
4. Store in `data/audio/` directory
5. Create a `data/metadata.json` with file names, source URLs, speaker names, and topics

### Ground Truth for Evaluation

For each audio file, manually create 3-4 labeled queries with expected results:

```json
{
  "query": "machine learning applications in healthcare",
  "expected_results": [
    {
      "file": "episode_03.wav",
      "speaker": "Dr. Smith",
      "timestamp_range": [124.5, 145.2],
      "relevant_text": "We've been applying deep learning to radiology..."
    }
  ]
}
```

Total: ~20 labeled query-result pairs across all files.

---

## 4. Project Structure

```
audio-rag/
├── docker-compose.yml          # Postgres + pgvector
├── pyproject.toml               # Dependencies (uv/poetry)
├── .env.example                 # API keys template
├── data/
│   ├── audio/                   # Raw audio files (gitignored, large)
│   ├── transcripts/             # JSON transcripts with timestamps
│   ├── metadata.json            # File metadata + speaker info
│   └── eval/
│       └── golden_queries.json  # Labeled query-result pairs
├── src/
│   ├── __init__.py
│   ├── config.py                # Settings, DB connection, model paths
│   ├── transcribe.py            # Whisper API transcription
│   ├── diarize.py               # pyannote speaker diarization
│   ├── align.py                 # Merge transcription + diarization → speaker turns
│   ├── chunk.py                 # Chunking logic (by speaker turn)
│   ├── embed.py                 # Sentence-transformer embedding generation
│   ├── db.py                    # Postgres/pgvector schema + CRUD
│   ├── index.py                 # Orchestrates: transcribe → diarize → align → chunk → embed → store
│   ├── search.py                # Hybrid search: keyword + semantic + RRF
│   └── cli.py                   # CLI entry point for indexing and searching
├── tests/
│   ├── test_search.py           # Recall@k evaluation against golden queries
│   └── test_pipeline.py         # Unit tests for individual pipeline stages
├── IMPLEMENTATION_PLAN.md       # This file
└── README.md                    # Solution writeup (submission)
```

---

## 5. Database Schema

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Audio files metadata
CREATE TABLE audio_files (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL UNIQUE,
    source_url  TEXT,
    duration_s  FLOAT,
    speaker_a   TEXT,
    speaker_b   TEXT,
    topic       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Speaker turn chunks (the searchable unit)
CREATE TABLE chunks (
    id              SERIAL PRIMARY KEY,
    audio_file_id   INT REFERENCES audio_files(id),
    speaker         TEXT NOT NULL,
    text            TEXT NOT NULL,
    start_time      FLOAT NOT NULL,       -- seconds from start
    end_time        FLOAT NOT NULL,
    embedding       vector(384),           -- all-MiniLM-L6-v2
    text_search     tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for hybrid search
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);
CREATE INDEX idx_chunks_text_search ON chunks USING gin (text_search);
CREATE INDEX idx_chunks_audio_file ON chunks (audio_file_id);
```

---

## 6. Pipeline Details

### 6.1 Transcription (transcribe.py)

- Use `faster-whisper` with the `base` or `small` model (good accuracy-to-speed tradeoff on CPU)
- Enable word-level timestamps: `model.transcribe(audio, word_timestamps=True)`
- This gives us word-level timestamps: `[{"word": "hello", "start": 0.0, "end": 0.5}, ...]`
- Store raw transcript JSON in `data/transcripts/`
- Fully local and free — no API key needed

### 6.2 Diarization (diarize.py)

- Load pyannote.audio pipeline: `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")`
- Requires a HuggingFace token (accept model terms once)
- Produces speaker segments: `[(start, end, "SPEAKER_00"), (start, end, "SPEAKER_01"), ...]`

### 6.3 Alignment (align.py)

- For each word from Whisper, find the overlapping pyannote segment
- Assign each word to a speaker based on maximum temporal overlap
- Group consecutive words by the same speaker into **speaker turns**
- Each turn becomes one chunk with: speaker label, concatenated text, start/end timestamps

### 6.4 Chunking (chunk.py)

- Primary strategy: one chunk per speaker turn
- **Long turn handling**: if a speaker turn exceeds ~500 tokens (~375 words), split at sentence boundaries while preserving the speaker label
- **Short turn merging**: optionally merge very short turns (< 10 words) from the same speaker if separated by brief interjections
- Each chunk stores: `audio_file_id`, `speaker`, `text`, `start_time`, `end_time`

### 6.5 Embedding (embed.py)

- Load `sentence-transformers/all-MiniLM-L6-v2` via `sentence_transformers` library
- Encode chunk text → 384-dim vector
- Batch encode for efficiency (batch size 32)
- Normalize vectors (L2) for cosine similarity

### 6.6 Indexing (index.py)

- Orchestrator that runs the full pipeline per audio file:
  `load audio → transcribe → diarize → align → chunk → embed → insert into Postgres`
- Idempotent: skip files already indexed (check by filename)
- Rebuild index after all inserts (`REINDEX`)

---

## 7. Hybrid Search Design

### 7.1 Keyword Search (BM25-style via Postgres FTS)

```python
def keyword_search(query: str, k: int = 20) -> list[dict]:
    sql = """
        SELECT id, ts_rank_cd(text_search, query) AS score
        FROM chunks, plainto_tsquery('english', %s) query
        WHERE text_search @@ query
        ORDER BY score DESC
        LIMIT %s
    """
    # Returns ranked list of (chunk_id, score)
```

### 7.2 Semantic Search (pgvector cosine similarity)

```python
def semantic_search(query_embedding: list[float], k: int = 20) -> list[dict]:
    sql = """
        SELECT id, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    # Returns ranked list of (chunk_id, score)
```

### 7.3 Reciprocal Rank Fusion (RRF)

```python
def rrf_merge(keyword_results, semantic_results, k_rrf: int = 60) -> list[dict]:
    """
    RRF score = Σ 1 / (k + rank_i)  for each result list where the doc appears.
    k_rrf is a constant (typically 60) that dampens the effect of high ranks.
    """
    scores = defaultdict(float)
    for rank, result in enumerate(keyword_results):
        scores[result["id"]] += 1.0 / (k_rrf + rank + 1)
    for rank, result in enumerate(semantic_results):
        scores[result["id"]] += 1.0 / (k_rrf + rank + 1)
    
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged
```

### 7.4 Search Output Format

Each result returns:
- **File**: which audio recording
- **Speaker**: who said it
- **Timestamp**: start and end time (formatted as MM:SS)
- **Text**: the matching transcript chunk
- **Score**: combined RRF score
- **Match type**: keyword, semantic, or both

---

## 8. Evaluation Strategy

### 8.1 Metrics

| Metric       | What It Measures                                                  | Target  |
|--------------|-------------------------------------------------------------------|---------|
| Recall@5     | Of relevant results, how many appear in the top 5?                | ≥ 0.70  |
| Recall@10    | Of relevant results, how many appear in the top 10?               | ≥ 0.85  |
| MRR          | Mean Reciprocal Rank — how high is the first relevant result?     | ≥ 0.60  |
| Precision@5  | Of the top 5 results, how many are actually relevant?             | ≥ 0.50  |

### 8.2 Test Structure

```python
# tests/test_search.py

def test_recall_at_k():
    """
    For each labeled query in golden_queries.json:
    1. Run hybrid search
    2. Check if expected chunks appear in top-k results
    3. Compute recall@k
    4. Assert aggregate recall meets threshold
    """

def test_keyword_only_recall():
    """Baseline: keyword search alone."""

def test_semantic_only_recall():
    """Baseline: semantic search alone."""

def test_hybrid_beats_individual():
    """Hybrid recall@k should be >= max(keyword, semantic) recall@k."""
```

### 8.3 Production Metrics to Consider

If brought to production, we'd additionally track:

- **Latency**: p50/p95 search response time (target < 200ms)
- **Indexing throughput**: minutes of audio indexed per minute of wall clock
- **Embedding drift**: monitor if embedding model updates change result quality
- **User satisfaction**: click-through rate on search results, result feedback signals
- **Freshness**: time from audio upload to searchable
- **Cost**: Whisper API cost per minute of audio transcribed

---

## 9. Implementation Order

### Phase 1: Foundation
1. Set up project scaffolding (`pyproject.toml`, directory structure, Docker Compose)
2. Set up Postgres + pgvector via Docker
3. Create database schema and connection utilities

### Phase 2: Data Pipeline
4. Download and prepare golden dataset (5-6 audio files)
5. Implement transcription via Whisper API
6. Implement diarization via pyannote.audio
7. Implement alignment (merge Whisper words + pyannote speakers)
8. Implement chunking by speaker turn

### Phase 3: Indexing
9. Implement embedding generation with sentence-transformers
10. Implement database insertion (chunks + embeddings + tsvectors)
11. Build the indexing orchestrator (end-to-end pipeline)

### Phase 4: Search
12. Implement keyword search (Postgres FTS)
13. Implement semantic search (pgvector)
14. Implement RRF fusion
15. Build CLI for search queries

### Phase 5: Evaluation
16. Create golden query set with labeled expected results
17. Implement recall@k, MRR, precision@k tests
18. Run evaluation, tune RRF constant and retrieval depth
19. Compare hybrid vs. keyword-only vs. semantic-only

### Phase 6: Documentation & Submission
20. Write README with design rationale, success criteria, results, limitations
21. Document agent collaboration (this plan + interaction traces)

---

## 10. Dependencies

```toml
[project]
name = "audio-rag"
requires-python = ">=3.11"

[project.dependencies]
faster-whisper = ">=1.0"      # Local Whisper transcription (free)
pyannote-audio = ">=3.1"      # Speaker diarization
sentence-transformers = ">=2.2" # Local embeddings
psycopg = {version = ">=3.1", extras = ["binary"]}  # Postgres driver
pgvector = ">=0.2"            # pgvector Python bindings
numpy = ">=1.24"
click = ">=8.0"               # CLI framework
python-dotenv = ">=1.0"       # .env file loading
pydub = ">=0.25"              # Audio format conversion
torch = ">=2.0"               # Required by pyannote

[project.optional-dependencies]
dev = [
    "pytest = >=7.0",
    "pytest-asyncio",
]
```

---

## 11. Decisions Log

| Decision                  | Choice                              | Notes                                              |
|---------------------------|-------------------------------------|----------------------------------------------------|
| Search interface          | CLI only (Click)                    | Focus effort on search quality, not UI             |
| Speaker name mapping      | Manual mapping in metadata.json     | Map SPEAKER_00/01 to real names after diarization  |
| Audio file storage        | Include files in repo               | Easier for reviewers to reproduce; use Git LFS if needed |

### Still Needed
- [ ] **HuggingFace token**: pyannote requires accepting model terms on HuggingFace. Need an HF account/token.
- [x] ~~OpenAI API key~~: No longer needed — using faster-whisper locally.

---

## 12. Success Criteria

| Criterion                                  | Definition of Done                                          |
|--------------------------------------------|-------------------------------------------------------------|
| Golden dataset complete                    | 5-6 audio files, 8-10 min each, 2 speakers, metadata.json  |
| Transcription works                        | Word-level timestamps for all files                         |
| Diarization works                          | Speaker labels correctly assigned to turns                  |
| Hybrid search returns results              | Both keyword and semantic paths produce ranked results      |
| Results include file, timestamp, speaker   | Each result shows source file, MM:SS range, and speaker     |
| Recall@5 ≥ 0.70 on golden queries          | Automated test passes                                      |
| Hybrid outperforms individual strategies   | Automated comparison test passes                            |
| Solution runs locally (except transcription)| Embeddings + search + DB all run without external APIs      |

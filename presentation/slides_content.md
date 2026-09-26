# Audio RAG: Hybrid Search Over Audio Transcripts
## Slide-by-slide content for pitch deck

---

### Slide 1: Title

**Audio RAG: Hybrid Search Over Audio Transcripts**

Effective retrieval from audio conversations using keyword + semantic search with speaker diarization

---

### Slide 2: The Problem

- Audio recordings contain valuable information but are unsearchable by default
- Transcripts alone aren't enough — users search using their own words, not the exact words spoken
- Need to know **who said what** and **when** — not just find matching text
- Challenge: Build a hybrid search system that handles both exact keyword queries and natural-language meaning queries

---

### Slide 3: Solution Overview

**A four-stage pipeline: Transcribe → Diarize → Index → Search**

- Transcribe audio with word-level timestamps (faster-whisper)
- Identify speakers using voice fingerprinting (pyannote)
- Align words to speakers, chunk by speaker turn, embed + index
- Search with keyword + semantic + Reciprocal Rank Fusion (RRF)

Every result returns: **File | Speaker | Timestamp | Text**

---

### Slide 4: Architecture Diagram

```
Audio Files (5 × 10 min, 2 speakers each)
       ↓
  faster-whisper          → words + timestamps
  pyannote.audio          → speaker segments
       ↓
  Alignment + Chunking    → speaker turns
       ↓
  all-MiniLM-L6-v2        → 384-dim embeddings
  Postgres tsvector        → full-text tokens
       ↓
  Postgres + pgvector      → unified storage
       ↓
  Hybrid Search (RRF)      → ranked results
```

All models run locally. Only dependency is Docker for Postgres.

---

### Slide 5: Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Chunking strategy | By speaker turn | Natural boundary, preserves who said what |
| Embedding model | all-MiniLM-L6-v2 | Fast on CPU, 384-dim, strong quality |
| Search fusion | Reciprocal Rank Fusion | Simple, effective, no tuning required |
| Database | Postgres + pgvector | Vectors + full-text search in one database |
| Transcription | faster-whisper (local) | Free, accurate, word-level timestamps |
| Diarization | pyannote.audio (local) | State-of-the-art, open-source |

---

### Slide 6: Why Hybrid Search?

**Keyword search alone fails 61% of the time.**

Users ask: "what caused the fall of Constantinople"
Transcript says: "armies from mostly France sack Constantinople and dismember the empire"

- No exact word overlap → keyword search returns **nothing**
- Semantic search understands the meaning → finds the right chunk
- Hybrid combines both → best ranking when terms AND meaning match

---

### Slide 7: Evaluation Design

**36 golden queries across 3 categories:**

- **12 keyword queries** — exact words from transcript ("Sam Altman accused of murder")
- **15 semantic queries** — paraphrased, different wording ("are we living in a computer simulation")
- **8 hybrid queries** — partial overlap + meaning ("Epstein did not kill himself guards cameras")

**8 multi-result queries** testing cross-segment and cross-file retrieval

**Metrics:** Recall@k, MRR, Precision@k — evaluated per category to isolate each search modality's contribution

---

### Slide 8: Results

| Strategy | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| **Hybrid** | **0.929** | **0.957** | **0.858** |
| Keyword Only | 0.386 | 0.386 | 0.400 |
| Semantic Only | 0.929 | 0.957 | 0.772 |

All targets exceeded:
- Recall@5 = 0.929 (target ≥ 0.70) ✓
- Recall@10 = 0.957 (target ≥ 0.85) ✓
- MRR = 0.858 (target ≥ 0.60) ✓

---

### Slide 9: Per-Category Breakdown

**This is where hybrid search proves its value.**

| Strategy | Keyword Queries (12) | Semantic Queries (15) | Hybrid Queries (8) |
|---|---|---|---|
| Hybrid Recall@10 | 0.917 | 0.967 | **1.000** |
| Keyword Recall@10 | 0.625 | **0.133** | 0.500 |
| Semantic Recall@10 | 0.917 | 0.967 | 1.000 |

- Keyword search scores **0.133 recall on semantic queries** — near total failure
- Semantic search handles everything but ranks less precisely (MRR 0.772 vs 0.858)
- Hybrid achieves **perfect recall and MRR on hybrid-category queries**

---

### Slide 10: Key Observations

1. **Keyword search alone is insufficient** — returned NO RESULTS for 22/35 queries
2. **Semantic search drives recall** — hybrid and semantic have identical recall (0.957)
3. **Hybrid's advantage is ranking** — RRF boosts results that match both modalities, improving MRR by 11% over semantic-only
4. **Multi-result queries are harder** — recall drops from ~1.0 to 0.5 at k=5 for multi-expected-result queries
5. **One persistent failure** — "condolence cards" query fails across all strategies due to a chunking boundary issue, not a search problem

---

### Slide 11: Limitations and Future Work

**Limitations:**
- 36 queries / single annotator — directional, not statistically robust
- Precision@k is structurally low with 1-2 expected results per query
- Transcription errors propagate to both search modalities
- CPU-only: diarization takes ~15 min per 10-min file

**Future improvements:**
- Overlapping chunks for better boundary coverage
- Re-ranking with a cross-encoder for higher precision
- GPU acceleration for real-time indexing
- User feedback loop to improve ranking over time

---

### Slide 12: Tech Stack Summary

| Component | Tool | Runs |
|---|---|---|
| Transcription | faster-whisper (base) | Local, free |
| Diarization | pyannote.audio 3.1 | Local, free |
| Embeddings | all-MiniLM-L6-v2 | Local, free |
| Database | Postgres 16 + pgvector | Docker |
| Keyword search | Postgres full-text (tsvector) | Local |
| Semantic search | pgvector cosine similarity | Local |
| Fusion | Reciprocal Rank Fusion (RRF) | Python |
| CLI | Click | Python |

**Fully local. No paid APIs. Reproducible from a single `docker compose up`.**

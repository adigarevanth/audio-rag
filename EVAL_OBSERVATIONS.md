# Evaluation Observations

## 1. Evaluation Design Philosophy

Our evaluation tests whether hybrid search (keyword + semantic combined via Reciprocal Rank Fusion) delivers better retrieval quality than either strategy alone across diverse audio transcripts. The golden dataset was deliberately constructed to stress-test each search modality independently and in combination.

### 1.1 Dataset Composition

- **5 audio files** covering distinct domains: AI/technology (Andrew Ng), ancient history (Lex Fridman), current events (Joe Rogan/Elon Musk), psychology/relationships (Raj Shamani), and career/finance (Finance with Sharan)
- **36 labeled queries** with **43 total expected results** (some queries expect multiple matching chunks)
- **8 queries with multiple expected results**, testing whether the system can surface more than one relevant segment for a broad question

### 1.2 Query Category Design

Every query was manually classified into one of three categories to test each search modality's strengths and weaknesses:

| Category | Count | Purpose | Example |
|---|---|---|---|
| **keyword** | 12 | Contains exact words from transcript. Should be found by keyword search. | "Sam Altman accused of murder" |
| **semantic** | 15 | Uses different words but same meaning. Should only be found by semantic search. | "are we living in a computer simulation" (transcript says "simulation is more and more undeniable") |
| **hybrid** | 8 | Benefits from both: partial keyword overlap plus meaning. | "Epstein did not kill himself guards cameras" |

This categorization lets us verify a key hypothesis: keyword search should excel at keyword queries but fail on semantic ones; semantic search should handle both but rank less precisely; hybrid should perform well across all categories.

### 1.3 Key Decisions in Test Case Design

**Why these specific queries?**
- Queries were crafted to span a spectrum from exact-match (proper nouns like "Errol Flynn", "Sam Altman") to fully paraphrased ("what would happen if a giant space rock hit Earth" instead of "asteroid impact extinction")
- Cross-file queries like "what career advice would you give to someone starting out in 2026" deliberately expect results from two different audio files (Finance with Sharan + Andrew Ng), testing whether the system can surface relevant content across the entire corpus

**Why timestamp-overlap matching?**
- Our matching logic checks file name + timestamp overlap (with 5-second tolerance) rather than exact text matching
- This is more robust to chunking variations: if the same content gets split differently across runs, the evaluation still works as long as the correct time range is returned

**Why manual labeling?**
- Each expected result was created by reading the actual transcript output, identifying which chunks contain the answer, and recording their file name and timestamp range
- This is labor-intensive but avoids circular evaluation (using the same embeddings to generate labels that the embeddings are tested against)

---

## 2. Metrics: What They Measure and Why They Matter

### 2.1 Recall@k

```
Recall@k = (expected results found in top k) / (total expected results)
```

**Why it matters:** The most critical metric for a search system. If the relevant chunk isn't in the results at all, no amount of good ranking helps. Recall@5 tests whether users find what they need without scrolling. Recall@10 tests whether the result exists anywhere in a reasonable result set.

**Our targets:** Recall@5 >= 0.70, Recall@10 >= 0.85

### 2.2 MRR (Mean Reciprocal Rank)

```
MRR = average of (1 / rank of first relevant result) across all queries
```

**Why it matters:** Measures ranking quality. Two systems can have identical recall but different MRR. A system that always puts the relevant result at rank 1 (MRR = 1.0) is far more useful than one that buries it at rank 5 (MRR = 0.2). This is where hybrid search is expected to outperform individual strategies.

**Our target:** MRR >= 0.60

### 2.3 Precision@k

```
Precision@k = (relevant results in top k) / k
```

**Why it matters:** Measures noise in the result set. High precision means fewer irrelevant results cluttering the output. However, precision is inherently limited by our evaluation design: most queries have only 1-2 expected results, so the theoretical maximum Precision@10 for a single-result query is 0.10.

**Interpretation note:** Low precision values (0.20-0.35) do not indicate poor search quality. They reflect the small number of labeled relevant chunks relative to k. Many "irrelevant" results are actually topically related but weren't labeled as expected results.

---

## 3. Results

### 3.1 Aggregate Performance

| Strategy | Recall@5 | Recall@10 | MRR | Precision@5 | Precision@10 |
|---|---|---|---|---|---|
| **Hybrid** | **0.929** | **0.957** | **0.858** | 0.349 | 0.234 |
| Keyword Only | 0.386 | 0.386 | 0.400 | 0.400 | 0.400 |
| Semantic Only | 0.929 | 0.957 | 0.772 | 0.331 | 0.234 |

**All primary targets met:**
- Recall@5 = 0.929 (target >= 0.70) ✓
- Recall@10 = 0.957 (target >= 0.85) ✓
- MRR = 0.858 (target >= 0.60) ✓

### 3.2 Performance by Query Category

This is the most revealing breakdown. It validates that hybrid search works as intended.

#### Hybrid Search (the system we ship)

| Category | Count | Recall@10 | MRR |
|---|---|---|---|
| keyword | 12 | 0.917 | 0.767 |
| semantic | 15 | 0.967 | 0.856 |
| hybrid | 8 | **1.000** | **1.000** |

Hybrid search performs well across all categories, with perfect scores on queries designed to benefit from both modalities.

#### Keyword Only Search

| Category | Count | Recall@10 | MRR |
|---|---|---|---|
| keyword | 12 | 0.625 | 0.667 |
| semantic | 15 | **0.133** | **0.133** |
| hybrid | 8 | 0.500 | 0.500 |

**Critical finding:** Keyword search fails catastrophically on semantic queries (Recall@10 = 0.133). This is expected and validates the query design. Queries like "what caused the fall of Constantinople", "how to be more attractive and charismatic", or "will AI replace all jobs" use different words than the transcript, so full-text search returns NO RESULTS.

Even on keyword-category queries, keyword search only achieves 0.625 recall. Some keyword queries use terms that appear in the transcript but in different grammatical forms that Postgres full-text search stemming doesn't reconcile.

#### Semantic Only Search

| Category | Count | Recall@10 | MRR |
|---|---|---|---|
| keyword | 12 | 0.917 | 0.683 |
| semantic | 15 | 0.967 | 0.772 |
| hybrid | 8 | 1.000 | 0.906 |

Semantic search matches hybrid on recall but has consistently lower MRR. It finds the right chunks but doesn't always rank them first. The MRR gap is most visible on keyword queries (0.683 vs 0.767) where exact term matching gives hybrid an additional ranking signal.

### 3.3 Key Observations

**Observation 1: Hybrid search produces the best MRR across all categories.**

The RRF fusion boosts results that appear in both keyword and semantic result lists. When a chunk matches both on exact terms and meaning, it gets a higher combined score and ranks first. This is why hybrid MRR (0.858) exceeds both keyword (0.400) and semantic (0.772).

**Observation 2: Keyword search alone is insufficient for audio transcript search.**

Keyword search returned NO RESULTS for 22 out of 35 queries (61%). People naturally search using their own words, not the exact words spoken in a recording. This validates the need for semantic search.

**Observation 3: Semantic search is the primary driver of recall.**

Hybrid and semantic have identical Recall@5 (0.929) and Recall@10 (0.957). The embedding model successfully surfaces relevant chunks even when query wording differs entirely from transcript text. The added value of keyword search is in ranking, not retrieval.

**Observation 4: Multi-result queries are harder.**

Queries with 2 expected results had lower scores than single-result queries. For example:
- "what would happen if a giant space rock hit Earth" (2 expected, Recall@5 = 0.500): found one expected chunk but not the second
- "what career advice would you give to someone starting out in 2026" (2 expected, cross-file): found both in top 10 but only one in top 5

This suggests that recall improves further at higher k values and that multi-result queries are a more demanding (and realistic) test.

**Observation 5: One persistent failure across all strategies.**

The query "condolence cards sales strategy market share" scored 0.000 recall across all three strategies. Investigation shows the transcript chunk for this story is very short ("But he's still didn't understand how this condolence cards right") and doesn't contain the actual narrative about market share growth. The detailed story likely falls in an adjacent chunk with different wording. This is a chunking boundary issue, not a search failure.

---

## 4. Limitations

### 4.1 Golden Dataset Size
36 queries across 5 files is sufficient for directional validation but not statistically robust. A production evaluation would use hundreds of queries with multiple annotators to reduce labeling bias.

### 4.2 Single Annotator
All expected results were labeled by a single person (assisted by AI transcript analysis). Inter-annotator agreement is not measured. Different people might label different chunks as relevant for the same query.

### 4.3 Precision Interpretation
With 1-2 expected results per query, precision@k is structurally low and not meaningfully comparable across systems. A proper precision evaluation would require exhaustive relevance labeling of all chunks for each query.

### 4.4 Timestamp Tolerance
The 5-second tolerance in matching allows near-misses to count as hits. A stricter tolerance might reveal more alignment issues between the golden labels and actual chunk boundaries.

### 4.5 Transcription Quality
Whisper transcription errors (e.g., "pickle" instead of "nickel", "Heracus" instead of "Heraclius") affect both keyword and semantic search. The evaluation does not isolate transcription errors from search errors.

---

## 5. Production Considerations

If this system were deployed to production, we would additionally monitor:

| Metric | What It Measures | Target |
|---|---|---|
| **p50/p95 search latency** | Response time for end users | < 200ms / < 500ms |
| **Indexing throughput** | Minutes of audio processed per minute of wall clock | Track over time |
| **Embedding drift** | Whether model updates change result quality | Monitor recall on fixed test set |
| **User click-through rate** | Do users click the top result? | > 40% |
| **Query abandonment rate** | Do users reformulate or give up? | < 20% |
| **Zero-result rate** | How often does search return nothing useful? | < 5% |
| **Freshness** | Time from audio upload to searchable | < 30 min |

---

## 6. Summary

The hybrid search system achieves **Recall@10 = 0.957** and **MRR = 0.858**, exceeding all targets. The per-category analysis validates the architectural decision to combine keyword and semantic search: keyword search alone fails on 61% of queries, while hybrid search maintains high recall across all query types and produces the best ranking through RRF fusion. The primary value of keyword search is improving the ranking of results that also match semantically, not expanding the set of retrievable results.

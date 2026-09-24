"""Hybrid search: keyword (Postgres FTS) + semantic (pgvector) + RRF fusion."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import psycopg

from src.config import RRF_K, SEARCH_K
from src.embed import embed_query

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with metadata."""

    chunk_id: int
    filename: str
    speaker: str
    text: str
    start_time: float
    end_time: float
    score: float
    match_type: str  # "keyword", "semantic", or "both"

    @property
    def timestamp(self) -> str:
        """Format start-end as MM:SS - MM:SS."""
        def fmt(s: float) -> str:
            m, sec = divmod(int(s), 60)
            return f"{m:02d}:{sec:02d}"
        return f"{fmt(self.start_time)} - {fmt(self.end_time)}"


def keyword_search(
    conn: psycopg.Connection,
    query: str,
    k: int = SEARCH_K,
) -> list[tuple[int, float]]:
    """
    Full-text keyword search using Postgres tsvector/tsquery.

    Returns list of (chunk_id, score) ordered by relevance.
    """
    rows = conn.execute(
        """
        SELECT id, ts_rank_cd(text_search, query) AS score
        FROM chunks, plainto_tsquery('english', %s) query
        WHERE text_search @@ query
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, k),
    ).fetchall()
    return [(r["id"], r["score"]) for r in rows]


def semantic_search(
    conn: psycopg.Connection,
    query_embedding: np.ndarray,
    k: int = SEARCH_K,
) -> list[tuple[int, float]]:
    """
    Semantic search using pgvector cosine similarity.

    Returns list of (chunk_id, score) ordered by similarity.
    """
    rows = conn.execute(
        """
        SELECT id, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding.tolist(), query_embedding.tolist(), k),
    ).fetchall()
    return [(r["id"], r["score"]) for r in rows]


def rrf_merge(
    *result_lists: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion across multiple ranked result lists.

    RRF score for a document = sum of 1/(k + rank) across all lists
    where the document appears.

    Args:
        result_lists: One or more lists of (chunk_id, original_score).
        k: Damping constant (default 60).

    Returns:
        Merged list of (chunk_id, rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = defaultdict(float)

    for results in result_lists:
        for rank, (chunk_id, _) in enumerate(results):
            scores[chunk_id] += 1.0 / (k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    keyword_k: int = SEARCH_K,
    semantic_k: int = SEARCH_K,
) -> list[SearchResult]:
    """
    Run hybrid search: keyword + semantic + RRF fusion.

    Args:
        conn: Database connection.
        query: The search query string.
        k: Number of final results to return.
        keyword_k: Number of keyword results to fetch.
        semantic_k: Number of semantic results to fetch.

    Returns:
        List of SearchResult objects, ranked by RRF score.
    """
    # Run both searches
    kw_results = keyword_search(conn, query, k=keyword_k)
    query_emb = embed_query(query)
    sem_results = semantic_search(conn, query_emb, k=semantic_k)

    logger.info(
        "Keyword: %d results, Semantic: %d results",
        len(kw_results), len(sem_results),
    )

    # Track which IDs came from which source
    kw_ids = {cid for cid, _ in kw_results}
    sem_ids = {cid for cid, _ in sem_results}

    # Merge with RRF
    merged = rrf_merge(kw_results, sem_results)
    top_ids = [cid for cid, _ in merged[:k]]
    top_scores = {cid: score for cid, score in merged[:k]}

    # Fetch full details
    from src.db import get_chunk_details
    details = get_chunk_details(conn, top_ids)

    # Build results
    results: list[SearchResult] = []
    for row in details:
        cid = row["id"]
        in_kw = cid in kw_ids
        in_sem = cid in sem_ids
        if in_kw and in_sem:
            match_type = "both"
        elif in_kw:
            match_type = "keyword"
        else:
            match_type = "semantic"

        results.append(SearchResult(
            chunk_id=cid,
            filename=row["filename"],
            speaker=row["speaker"],
            text=row["text"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            score=top_scores[cid],
            match_type=match_type,
        ))

    return results


def keyword_only_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
) -> list[SearchResult]:
    """Run keyword-only search (for evaluation baselines)."""
    kw_results = keyword_search(conn, query, k=k)
    top_ids = [cid for cid, _ in kw_results]
    top_scores = {cid: score for cid, score in kw_results}

    from src.db import get_chunk_details
    details = get_chunk_details(conn, top_ids)

    return [
        SearchResult(
            chunk_id=row["id"],
            filename=row["filename"],
            speaker=row["speaker"],
            text=row["text"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            score=top_scores[row["id"]],
            match_type="keyword",
        )
        for row in details
    ]


def semantic_only_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
) -> list[SearchResult]:
    """Run semantic-only search (for evaluation baselines)."""
    query_emb = embed_query(query)
    sem_results = semantic_search(conn, query_emb, k=k)
    top_ids = [cid for cid, _ in sem_results]
    top_scores = {cid: score for cid, score in sem_results}

    from src.db import get_chunk_details
    details = get_chunk_details(conn, top_ids)

    return [
        SearchResult(
            chunk_id=row["id"],
            filename=row["filename"],
            speaker=row["speaker"],
            text=row["text"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            score=top_scores[row["id"]],
            match_type="semantic",
        )
        for row in details
    ]

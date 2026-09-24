"""Postgres + pgvector database operations."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from src.chunk import Chunk
from src.config import DATABASE_URL

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Get a database connection with pgvector registered."""
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def insert_audio_file(
    conn: psycopg.Connection,
    filename: str,
    source_url: str | None = None,
    duration_s: float | None = None,
    speaker_a: str | None = None,
    speaker_b: str | None = None,
    topic: str | None = None,
) -> int:
    """Insert an audio file record and return its ID."""
    row = conn.execute(
        """
        INSERT INTO audio_files (filename, source_url, duration_s, speaker_a, speaker_b, topic)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE SET
            source_url = EXCLUDED.source_url,
            duration_s = EXCLUDED.duration_s,
            speaker_a = EXCLUDED.speaker_a,
            speaker_b = EXCLUDED.speaker_b,
            topic = EXCLUDED.topic
        RETURNING id
        """,
        (filename, source_url, duration_s, speaker_a, speaker_b, topic),
    ).fetchone()
    conn.commit()
    return row["id"]


def insert_chunks(
    conn: psycopg.Connection,
    audio_file_id: int,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    speaker_map: dict[str, str] | None = None,
) -> int:
    """
    Insert chunks with embeddings into the database.

    Args:
        conn: Database connection.
        audio_file_id: FK to audio_files table.
        chunks: List of Chunk objects.
        embeddings: numpy array of shape (len(chunks), dim).
        speaker_map: Optional mapping from pyannote labels (SPEAKER_00)
                     to real names (e.g. {"SPEAKER_00": "Lex Fridman"}).

    Returns:
        Number of chunks inserted.
    """
    speaker_map = speaker_map or {}

    # Delete existing chunks for this audio file (idempotent re-indexing)
    conn.execute("DELETE FROM chunks WHERE audio_file_id = %s", (audio_file_id,))

    count = 0
    for chunk, embedding in zip(chunks, embeddings):
        speaker = speaker_map.get(chunk.speaker, chunk.speaker)
        conn.execute(
            """
            INSERT INTO chunks (audio_file_id, speaker, text, start_time, end_time, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
            """,
            (
                audio_file_id,
                speaker,
                chunk.text,
                chunk.start_time,
                chunk.end_time,
                embedding.tolist(),
            ),
        )
        count += 1

    conn.commit()
    logger.info("Inserted %d chunks for audio_file_id=%d", count, audio_file_id)
    return count


def ensure_vector_index(conn: psycopg.Connection) -> None:
    """
    Create the IVFFlat index on embeddings if it doesn't exist.

    Should be called after data is loaded (IVFFlat needs rows to train).
    """
    # Check if index exists
    exists = conn.execute(
        "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_embedding'"
    ).fetchone()

    if not exists:
        row_count = conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()
        if row_count and row_count["cnt"] > 0:
            lists = min(50, max(1, row_count["cnt"] // 10))
            logger.info("Creating IVFFlat index with lists=%d...", lists)
            conn.execute(
                f"""
                CREATE INDEX idx_chunks_embedding
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists})
                """
            )
            conn.commit()
            logger.info("IVFFlat index created.")
        else:
            logger.warning("No chunks in database; skipping index creation.")


def get_audio_file_id(conn: psycopg.Connection, filename: str) -> int | None:
    """Look up an audio file ID by filename."""
    row = conn.execute(
        "SELECT id FROM audio_files WHERE filename = %s", (filename,)
    ).fetchone()
    return row["id"] if row else None


def get_chunk_details(conn: psycopg.Connection, chunk_ids: list[int]) -> list[dict]:
    """Fetch full chunk details by IDs, preserving order."""
    if not chunk_ids:
        return []
    placeholders = ",".join(["%s"] * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT c.id, c.speaker, c.text, c.start_time, c.end_time,
               a.filename
        FROM chunks c
        JOIN audio_files a ON c.audio_file_id = a.id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()

    # Preserve the requested order
    row_map = {r["id"]: r for r in rows}
    return [row_map[cid] for cid in chunk_ids if cid in row_map]

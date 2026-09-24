"""Orchestrate the full indexing pipeline: transcribe → diarize → align → chunk → embed → store."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.align import align_transcript_with_speakers
from src.chunk import create_chunks
from src.config import AUDIO_DIR, METADATA_PATH
from src.db import (
    ensure_vector_index,
    get_audio_file_id,
    get_connection,
    insert_audio_file,
    insert_chunks,
)
from src.diarize import diarize_audio, load_diarization, save_diarization
from src.embed import embed_chunks
from src.transcribe import load_transcript, save_transcript, transcribe_audio

logger = logging.getLogger(__name__)


def load_metadata() -> dict:
    """Load the metadata.json file describing the golden dataset."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}.\n"
            "Create data/metadata.json with audio file info."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def index_audio_file(
    audio_path: Path,
    file_meta: dict,
    force: bool = False,
) -> None:
    """
    Run the full indexing pipeline for a single audio file.

    Steps:
        1. Transcribe (or load cached transcript)
        2. Diarize (or load cached diarization)
        3. Align words with speaker segments
        4. Chunk by speaker turn
        5. Generate embeddings
        6. Store in Postgres

    Args:
        audio_path: Path to the audio file.
        file_meta: Metadata dict for this file from metadata.json.
        force: If True, re-process even if already indexed.
    """
    filename = audio_path.name

    with get_connection() as conn:
        existing_id = get_audio_file_id(conn, filename)
        if existing_id and not force:
            logger.info("'%s' already indexed (id=%d). Skipping. Use --force to re-index.", filename, existing_id)
            return

    # Step 1: Transcribe
    transcript = load_transcript(filename)
    if transcript and not force:
        logger.info("Using cached transcript for '%s'", filename)
    else:
        transcript = transcribe_audio(audio_path)
        save_transcript(transcript)

    # Step 2: Diarize
    diarization = load_diarization(filename)
    if diarization and not force:
        logger.info("Using cached diarization for '%s'", filename)
    else:
        num_speakers = file_meta.get("num_speakers", 2)
        diarization = diarize_audio(audio_path, num_speakers=num_speakers)
        save_diarization(diarization)

    # Step 3: Align
    aligned_words = align_transcript_with_speakers(transcript, diarization)

    # Step 4: Chunk
    chunks = create_chunks(aligned_words)
    if not chunks:
        logger.warning("No chunks created for '%s'. Skipping.", filename)
        return

    # Step 5: Embed
    embeddings = embed_chunks(chunks)

    # Step 6: Store in DB
    speaker_map = file_meta.get("speaker_map", {})

    with get_connection() as conn:
        audio_file_id = insert_audio_file(
            conn,
            filename=filename,
            source_url=file_meta.get("source_url"),
            duration_s=file_meta.get("duration_s"),
            speaker_a=speaker_map.get("SPEAKER_00", "Speaker A"),
            speaker_b=speaker_map.get("SPEAKER_01", "Speaker B"),
            topic=file_meta.get("topic"),
        )
        insert_chunks(conn, audio_file_id, chunks, embeddings, speaker_map)

    logger.info("Successfully indexed '%s' (%d chunks)", filename, len(chunks))


def index_all(force: bool = False) -> None:
    """
    Index all audio files listed in metadata.json.

    After indexing all files, creates the IVFFlat vector index.
    """
    metadata = load_metadata()

    for file_meta in metadata["files"]:
        filename = file_meta["filename"]
        audio_path = AUDIO_DIR / filename

        if not audio_path.exists():
            logger.error("Audio file not found: %s", audio_path)
            continue

        try:
            index_audio_file(audio_path, file_meta, force=force)
        except Exception:
            logger.exception("Failed to index '%s'", filename)

    # Create vector index after all data is loaded
    with get_connection() as conn:
        ensure_vector_index(conn)

    logger.info("Indexing complete.")

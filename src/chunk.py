"""Chunk aligned words into speaker turns for indexing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.align import AlignedWord
from src.config import MAX_CHUNK_WORDS, MIN_CHUNK_WORDS

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A searchable chunk representing a speaker turn or portion of one."""

    speaker: str
    text: str
    start_time: float
    end_time: float


def _split_at_sentence_boundaries(text: str, max_words: int) -> list[str]:
    """
    Split text into pieces of at most max_words, breaking at sentence
    boundaries (periods, question marks, exclamation marks).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces: list[str] = []
    current: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_word_count + sentence_words > max_words and current:
            pieces.append(" ".join(current))
            current = [sentence]
            current_word_count = sentence_words
        else:
            current.append(sentence)
            current_word_count += sentence_words

    if current:
        pieces.append(" ".join(current))

    return pieces


def create_chunks(
    aligned_words: list[AlignedWord],
    max_words: int = MAX_CHUNK_WORDS,
    min_words: int = MIN_CHUNK_WORDS,
) -> list[Chunk]:
    """
    Group aligned words into chunks by speaker turn.

    Primary strategy: one chunk per contiguous run of words from the same
    speaker. Long turns are split at sentence boundaries. Very short adjacent
    turns from the same speaker are merged.

    Args:
        aligned_words: Words with speaker labels from the alignment step.
        max_words: Maximum words per chunk before splitting.
        min_words: Minimum words; short same-speaker turns get merged.

    Returns:
        List of Chunk objects ready for embedding and indexing.
    """
    if not aligned_words:
        return []

    # Step 1: Group consecutive words by speaker into raw turns
    raw_turns: list[list[AlignedWord]] = []
    current_turn: list[AlignedWord] = [aligned_words[0]]

    for word in aligned_words[1:]:
        if word.speaker == current_turn[-1].speaker:
            current_turn.append(word)
        else:
            raw_turns.append(current_turn)
            current_turn = [word]
    raw_turns.append(current_turn)

    # Step 2: Merge very short adjacent turns from the same speaker
    merged_turns: list[list[AlignedWord]] = [raw_turns[0]]
    for turn in raw_turns[1:]:
        prev = merged_turns[-1]
        if (
            turn[0].speaker == prev[0].speaker
            and len(prev) < min_words
        ):
            # Merge with previous turn from same speaker
            merged_turns[-1] = prev + turn
        else:
            merged_turns.append(turn)

    # Step 3: Convert turns to Chunks, splitting long ones
    chunks: list[Chunk] = []
    for turn in merged_turns:
        speaker = turn[0].speaker
        text = " ".join(w.word for w in turn)
        start_time = turn[0].start
        end_time = turn[-1].end
        word_count = len(turn)

        if word_count <= max_words:
            chunks.append(Chunk(
                speaker=speaker,
                text=text,
                start_time=start_time,
                end_time=end_time,
            ))
        else:
            # Split long turns at sentence boundaries
            pieces = _split_at_sentence_boundaries(text, max_words)
            # Distribute timestamps proportionally
            total_duration = end_time - start_time
            total_words = word_count
            offset = start_time
            for piece in pieces:
                piece_words = len(piece.split())
                piece_duration = (piece_words / total_words) * total_duration
                chunks.append(Chunk(
                    speaker=speaker,
                    text=piece,
                    start_time=round(offset, 3),
                    end_time=round(offset + piece_duration, 3),
                ))
                offset += piece_duration

    logger.info("Created %d chunks from %d raw turns", len(chunks), len(raw_turns))
    return chunks

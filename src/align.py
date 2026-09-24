"""Align Whisper word timestamps with pyannote speaker segments."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.diarize import DiarizationResult, SpeakerSegment
from src.transcribe import TranscriptionResult, WordTimestamp

logger = logging.getLogger(__name__)


@dataclass
class AlignedWord:
    """A word assigned to a speaker."""

    word: str
    start: float
    end: float
    speaker: str


def _find_speaker(
    word: WordTimestamp,
    segments: list[SpeakerSegment],
) -> str:
    """
    Find which speaker segment a word belongs to.

    Uses maximum temporal overlap: for each segment that overlaps the word's
    time range, compute the overlap duration; pick the segment with the
    largest overlap. Falls back to "UNKNOWN" if no overlap is found.
    """
    word_mid = (word.start + word.end) / 2.0
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for seg in segments:
        # Quick check: is there any overlap?
        overlap_start = max(word.start, seg.start)
        overlap_end = min(word.end, seg.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg.speaker
        # If no overlap at all, check if the word midpoint falls in the segment
        elif overlap == 0.0 and best_overlap == 0.0:
            if seg.start <= word_mid <= seg.end:
                best_speaker = seg.speaker

    return best_speaker


def align_transcript_with_speakers(
    transcription: TranscriptionResult,
    diarization: DiarizationResult,
) -> list[AlignedWord]:
    """
    Merge word-level timestamps from Whisper with speaker segments from pyannote.

    For each word, finds the overlapping speaker segment and assigns the
    speaker label to that word.

    Args:
        transcription: Whisper transcription result with word timestamps.
        diarization: pyannote diarization result with speaker segments.

    Returns:
        List of words with speaker labels assigned.
    """
    segments = sorted(diarization.segments, key=lambda s: s.start)

    aligned: list[AlignedWord] = []
    for word in transcription.words:
        speaker = _find_speaker(word, segments)
        aligned.append(AlignedWord(
            word=word.word,
            start=word.start,
            end=word.end,
            speaker=speaker,
        ))

    # Log alignment stats
    speakers = set(aw.speaker for aw in aligned)
    unknown_count = sum(1 for aw in aligned if aw.speaker == "UNKNOWN")
    logger.info(
        "Aligned %d words to speakers %s (%d unassigned)",
        len(aligned), sorted(speakers - {"UNKNOWN"}), unknown_count,
    )

    return aligned

"""Transcribe audio files using faster-whisper (local, free)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from src.config import TRANSCRIPTS_DIR, WHISPER_MODEL

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """A single word with its start and end time in seconds."""

    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    """Full transcription result for one audio file."""

    audio_file: str
    language: str
    words: list[WordTimestamp]


def transcribe_audio(audio_path: Path, model_size: str | None = None) -> TranscriptionResult:
    """
    Transcribe an audio file and return word-level timestamps.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        model_size: Whisper model size override (tiny/base/small/medium/large-v3).

    Returns:
        TranscriptionResult with word-level timestamps.
    """
    model_size = model_size or WHISPER_MODEL
    logger.info("Loading faster-whisper model '%s'...", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    logger.info("Transcribing '%s'...", audio_path.name)
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,  # filter out non-speech segments
    )

    words: list[WordTimestamp] = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append(WordTimestamp(
                    word=w.word.strip(),
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                ))

    result = TranscriptionResult(
        audio_file=audio_path.name,
        language=info.language,
        words=words,
    )

    logger.info(
        "Transcribed %d words from '%s' (language: %s)",
        len(words), audio_path.name, info.language,
    )
    return result


def save_transcript(result: TranscriptionResult) -> Path:
    """Save transcription result as JSON to the transcripts directory."""
    stem = Path(result.audio_file).stem
    out_path = TRANSCRIPTS_DIR / f"{stem}_transcript.json"
    data = {
        "audio_file": result.audio_file,
        "language": result.language,
        "words": [asdict(w) for w in result.words],
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved transcript to '%s'", out_path)
    return out_path


def load_transcript(audio_filename: str) -> TranscriptionResult | None:
    """Load a previously saved transcript from disk."""
    stem = Path(audio_filename).stem
    path = TRANSCRIPTS_DIR / f"{stem}_transcript.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return TranscriptionResult(
        audio_file=data["audio_file"],
        language=data["language"],
        words=[WordTimestamp(**w) for w in data["words"]],
    )

"""Speaker diarization using pyannote.audio (local, free with HF token)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from pyannote.audio import Pipeline

from src.config import HF_TOKEN, TRANSCRIPTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A time range attributed to a specific speaker."""

    speaker: str
    start: float
    end: float


@dataclass
class DiarizationResult:
    """Full diarization result for one audio file."""

    audio_file: str
    segments: list[SpeakerSegment]


def diarize_audio(audio_path: Path, num_speakers: int = 2) -> DiarizationResult:
    """
    Run speaker diarization on an audio file.

    Args:
        audio_path: Path to the audio file.
        num_speakers: Expected number of speakers (default 2 for our dataset).

    Returns:
        DiarizationResult with speaker-labeled time segments.
    """
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is not set. pyannote.audio requires a HuggingFace token.\n"
            "1. Create a free account at https://huggingface.co\n"
            "2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Accept terms at https://huggingface.co/pyannote/segmentation-3.0\n"
            "4. Set HF_TOKEN in your .env file"
        )

    logger.info("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )

    logger.info("Diarizing '%s' (expecting %d speakers)...", audio_path.name, num_speakers)
    diarization = pipeline(str(audio_path), num_speakers=num_speakers)

    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            speaker=speaker,
            start=round(turn.start, 3),
            end=round(turn.end, 3),
        ))

    logger.info(
        "Diarization complete: %d segments, speakers: %s",
        len(segments),
        sorted(set(s.speaker for s in segments)),
    )
    return DiarizationResult(audio_file=audio_path.name, segments=segments)


def save_diarization(result: DiarizationResult) -> Path:
    """Save diarization result as JSON."""
    stem = Path(result.audio_file).stem
    out_path = TRANSCRIPTS_DIR / f"{stem}_diarization.json"
    data = {
        "audio_file": result.audio_file,
        "segments": [asdict(s) for s in result.segments],
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved diarization to '%s'", out_path)
    return out_path


def load_diarization(audio_filename: str) -> DiarizationResult | None:
    """Load a previously saved diarization from disk."""
    stem = Path(audio_filename).stem
    path = TRANSCRIPTS_DIR / f"{stem}_diarization.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return DiarizationResult(
        audio_file=data["audio_file"],
        segments=[SpeakerSegment(**s) for s in data["segments"]],
    )

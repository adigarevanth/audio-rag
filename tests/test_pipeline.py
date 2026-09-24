"""Unit tests for individual pipeline stages."""

from __future__ import annotations

import pytest

from src.align import AlignedWord, align_transcript_with_speakers
from src.chunk import Chunk, create_chunks
from src.diarize import DiarizationResult, SpeakerSegment
from src.transcribe import TranscriptionResult, WordTimestamp


class TestAlignment:
    """Test merging Whisper words with pyannote speaker segments."""

    def test_basic_alignment(self):
        """Words should be assigned to the correct speaker based on time overlap."""
        transcription = TranscriptionResult(
            audio_file="test.wav",
            language="en",
            words=[
                WordTimestamp(word="Hello", start=0.0, end=0.5),
                WordTimestamp(word="world", start=0.5, end=1.0),
                WordTimestamp(word="How", start=2.0, end=2.3),
                WordTimestamp(word="are", start=2.3, end=2.5),
                WordTimestamp(word="you", start=2.5, end=3.0),
            ],
        )
        diarization = DiarizationResult(
            audio_file="test.wav",
            segments=[
                SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=1.5),
                SpeakerSegment(speaker="SPEAKER_01", start=1.8, end=3.5),
            ],
        )

        aligned = align_transcript_with_speakers(transcription, diarization)

        assert len(aligned) == 5
        assert aligned[0].speaker == "SPEAKER_00"
        assert aligned[1].speaker == "SPEAKER_00"
        assert aligned[2].speaker == "SPEAKER_01"
        assert aligned[3].speaker == "SPEAKER_01"
        assert aligned[4].speaker == "SPEAKER_01"

    def test_unknown_speaker_for_gap(self):
        """Words not covered by any speaker segment get UNKNOWN."""
        transcription = TranscriptionResult(
            audio_file="test.wav",
            language="en",
            words=[
                WordTimestamp(word="gap", start=5.0, end=5.5),
            ],
        )
        diarization = DiarizationResult(
            audio_file="test.wav",
            segments=[
                SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
            ],
        )

        aligned = align_transcript_with_speakers(transcription, diarization)
        assert aligned[0].speaker == "UNKNOWN"


class TestChunking:
    """Test chunking aligned words into speaker turns."""

    def _make_words(self, texts_and_speakers: list[tuple[str, str]]) -> list[AlignedWord]:
        """Helper: create AlignedWord list from (text, speaker) pairs."""
        words = []
        t = 0.0
        for text, speaker in texts_and_speakers:
            for w in text.split():
                words.append(AlignedWord(word=w, start=t, end=t + 0.3, speaker=speaker))
                t += 0.4
        return words

    def test_basic_chunking(self):
        """Consecutive words from the same speaker form one chunk."""
        words = self._make_words([
            ("Hello how are you", "SPEAKER_00"),
            ("I am fine thanks", "SPEAKER_01"),
        ])

        chunks = create_chunks(words)

        assert len(chunks) == 2
        assert chunks[0].speaker == "SPEAKER_00"
        assert "Hello how are you" in chunks[0].text
        assert chunks[1].speaker == "SPEAKER_01"

    def test_long_turn_split(self):
        """Turns exceeding max_words should be split."""
        long_text = " ".join(["word"] * 50)
        words = self._make_words([(long_text, "SPEAKER_00")])

        chunks = create_chunks(words, max_words=20)

        assert len(chunks) > 1
        for c in chunks:
            assert c.speaker == "SPEAKER_00"

    def test_empty_input(self):
        """Empty word list produces no chunks."""
        chunks = create_chunks([])
        assert chunks == []

    def test_single_speaker(self):
        """All words from one speaker produce a single chunk."""
        words = self._make_words([
            ("One two three four five", "SPEAKER_00"),
        ])

        chunks = create_chunks(words)
        assert len(chunks) == 1
        assert chunks[0].speaker == "SPEAKER_00"

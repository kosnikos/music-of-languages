"""Pure data + measurement for the source probe (no network, no models)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from musiclang.clean.vad import extract_speech, total_speech_seconds
from musiclang.config import TARGET_SAMPLE_RATE

MIN_CLEAN_SPEECH_S: float = 30.0


@dataclass(frozen=True)
class RecordingRef:
    source: str       # 'radio' | 'podcast' | 'corpus'
    language: str     # seed-language key, e.g. 'finnish'
    channel_id: str   # station / show / speaker id — for distinctness counting
    kind: str         # 'progressive' | 'hls' | 'rss' | 'corpus'
    ref: str          # stream / m3u8 / enclosure url, or local wav path (corpus)


@dataclass(frozen=True)
class ProbeResult:
    source: str
    language: str
    channel_id: str
    kind: str
    capturable: bool
    clean_speech_s: float
    meets_30s: bool
    error: str = ""


def measure_cleanliness(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, *, speech_fn=extract_speech
) -> tuple[float, bool]:
    """Seconds of clean speech in `signal` and whether it clears the 30 s bar.

    `speech_fn(signal, sr) -> list[(start_s, end_s)]` is injectable so unit tests
    avoid loading the silero model; production passes the real `extract_speech`.
    """
    segments = speech_fn(signal, sr)
    clean_s = total_speech_seconds(segments)
    return clean_s, clean_s >= MIN_CLEAN_SPEECH_S

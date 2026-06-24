"""Voice-activity detection: extract speech spans with silero-vad.

Pure span helpers are unit-tested; `extract_speech` wraps the model (integration).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE


def merge_segments(
    segments: list[tuple[float, float]], gap: float = 0.2
) -> list[tuple[float, float]]:
    """Merge (start, end) spans (seconds) separated by less than `gap`."""
    if not segments:
        return []
    ordered = sorted(segments)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def total_speech_seconds(segments: list[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in segments))


def concat_speech(
    signal: np.ndarray, segments: list[tuple[float, float]], sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Concatenate the samples inside `segments` into a single array."""
    parts = [signal[int(start * sr):int(end * sr)] for start, end in segments]
    if not parts:
        return np.zeros(0, dtype=signal.dtype)
    return np.concatenate(parts).astype(signal.dtype)


@lru_cache(maxsize=1)
def _load_model():
    # Imported lazily so unit tests of the pure helpers don't need torch.
    from silero_vad import load_silero_vad

    return load_silero_vad()


def extract_speech(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, min_speech_s: float = 0.5
) -> list[tuple[float, float]]:
    """Return merged speech spans (seconds) detected by silero-vad."""
    import torch
    from silero_vad import get_speech_timestamps

    model = _load_model()
    tensor = torch.from_numpy(np.ascontiguousarray(signal))
    stamps = get_speech_timestamps(
        tensor, model, sampling_rate=sr, return_seconds=True,
        min_speech_duration_ms=int(min_speech_s * 1000),
    )
    segs = [(float(s["start"]), float(s["end"])) for s in stamps]
    return merge_segments(segs, gap=0.2)

"""Audio loading and loudness normalization."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE


def load_audio(path: str | Path, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at `sr` Hz, range [-1, 1]."""
    signal, _ = librosa.load(str(path), sr=sr, mono=True)
    return signal.astype(np.float32)


def normalize_loudness(signal: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale `signal` to `target_rms`. Silent signals are returned unchanged."""
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return signal.astype(np.float32)
    return (signal * (target_rms / rms)).astype(np.float32)

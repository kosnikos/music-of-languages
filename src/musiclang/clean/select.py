"""Select ONE analysis segment per recording: the first fixed-length window of
the already-cleaned, concatenated speech. One segment per recording keeps
segments independent (no within-recording pseudoreplication)."""

from __future__ import annotations

import numpy as np

from musiclang.clean.window import window_signal
from musiclang.config import TARGET_SAMPLE_RATE

SEGMENT_LENGTH_S: float = 30.0


def select_segment(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, length_s: float = SEGMENT_LENGTH_S
) -> np.ndarray | None:
    """First `length_s`-second window of `signal`, or None if `signal` is shorter.

    `signal` is the cleaned, concatenated speech of one recording (see
    `pipeline.clean_clip`). Returns the first non-overlapping window so the
    choice is deterministic.
    """
    windows = window_signal(signal, sr, length_s)
    return windows[0].samples if windows else None

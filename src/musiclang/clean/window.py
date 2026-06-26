"""Split a cleaned speech signal into fixed-length analysis windows.

Non-overlapping by default so sub-clips of one recording stay maximally
independent (overlap deepens autocorrelation). Window provenance feeds the
hierarchical sub-clip -> recording -> station aggregation used to avoid
pseudoreplication of within-recording variation.

Reference for fixed-window prosodic analysis of short clips: Deloche et al.
(2024), "Language identification from speech rhythm" — 10s clips suffice for
prosody-based language ID. arXiv:2401.14416
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Window(NamedTuple):
    start_s: float
    samples: np.ndarray


def window_signal(
    signal: np.ndarray,
    sr: int,
    length_s: float | None,
    hop_s: float | None = None,
    min_s: float | None = None,
) -> list[Window]:
    """Cut `signal` into windows of `length_s` seconds.

    `length_s=None` -> a single window spanning the whole signal (the "full"
    sweep value). Otherwise non-overlapping (`hop_s` defaults to `length_s`),
    dropping any trailing window shorter than `min_s` (default `length_s`).
    """
    n = len(signal)
    if n == 0:
        return []
    if length_s is None:
        return [Window(0.0, signal)]
    win = int(round(length_s * sr))
    hop = int(round((hop_s if hop_s is not None else length_s) * sr))
    min_len = win if min_s is None else int(round(min_s * sr))
    out: list[Window] = []
    start = 0
    while start < n:
        seg = signal[start : start + win]
        if len(seg) >= min_len:
            out.append(Window(start / sr, seg))
        start += hop
    return out

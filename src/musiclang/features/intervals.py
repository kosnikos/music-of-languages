"""Alignment-free vocalic/consonantal interval detection (Phase 0 approximation).

The vocalic/consonantal interval concept these feed (%V, ΔV, ΔC, PVIs) comes from
Ramus, Nespor & Mehler (1999): https://doi.org/10.1016/S0010-0277(99)00058-X
Our segmentation is an automatic approximation (voicing + relative intensity), inspired
by segmentation-free rhythm extraction (Galves et al. 2002); canonical studies segment
C/V intervals by hand or via forced alignment.
"""

from __future__ import annotations

import numpy as np
import parselmouth

from musiclang.config import TARGET_SAMPLE_RATE

_FRAME_STEP = 0.01      # 10 ms analysis step
_INTENSITY_PERCENTILE = 40  # frames below this percentile are treated as non-vocalic


def frames_to_intervals(
    is_vocalic: list[bool], frame_step: float
) -> tuple[list[float], list[float]]:
    """Convert a per-frame vocalic mask into vocalic and (medial) consonantal durations."""
    # Run-length encode the boolean mask.
    runs: list[tuple[bool, int]] = []
    for value in is_vocalic:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))

    vocalic = [length * frame_step for value, length in runs if value]

    # Consonantal = non-vocalic runs that sit strictly between two vocalic runs.
    consonantal: list[float] = []
    for i, (value, length) in enumerate(runs):
        if not value and 0 < i < len(runs) - 1:
            consonantal.append(length * frame_step)
    return vocalic, consonantal


def detect_intervals(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE
) -> tuple[list[float], list[float]]:
    """Detect vocalic/consonantal intervals via voicing + relative intensity.

    Reference: Ramus, Nespor & Mehler (1999), https://doi.org/10.1016/S0010-0277(99)00058-X
    """
    sound = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=_FRAME_STEP)
    intensity = sound.to_intensity(time_step=_FRAME_STEP)

    f0 = pitch.selected_array["frequency"]  # 0.0 where unvoiced
    pitch_times = pitch.xs()
    # Interpolate intensity to match pitch time grid (they may differ)
    intens_times = intensity.xs()
    intens_vals = intensity.values[0]
    intens = np.interp(pitch_times, intens_times, intens_vals, left=0.0, right=0.0)
    voiced = f0 > 0
    if voiced.any():
        threshold = np.percentile(intens[voiced], _INTENSITY_PERCENTILE)
    else:
        threshold = np.inf
    is_vocalic = [(bool(v) and i >= threshold) for v, i in zip(voiced, intens)]
    return frames_to_intervals(is_vocalic, frame_step=_FRAME_STEP)

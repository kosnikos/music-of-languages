"""F0 / intonation features via parselmouth.

F0 is estimated with Praat's autocorrelation pitch algorithm:
Boersma (1993), "Accurate short-term analysis of the fundamental frequency and the
harmonics-to-noise ratio of a sampled sound", Proceedings IFA 17:97-110,
https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf
Returned values are summary statistics of that F0 contour (mean/SD/min/max/range, plus
f0_slope = least-squares linear trend of F0 over time).
"""

from __future__ import annotations

import math

import numpy as np
import parselmouth

from musiclang.config import TARGET_SAMPLE_RATE


def pitch_features(signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> dict[str, float]:
    """Summary statistics of the F0 contour over voiced frames.

    Reference: Boersma (1993), https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf
    """
    keys = ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope"]
    sound = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.01)
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    voiced = f0 > 0
    if voiced.sum() < 2:
        return {k: math.nan for k in keys}

    fv = f0[voiced]
    tv = times[voiced]
    slope = float(np.polyfit(tv - tv[0], fv, 1)[0])  # Hz/s linear trend
    return {
        "f0_mean": float(np.mean(fv)),
        "f0_std": float(np.std(fv)),
        "f0_min": float(np.min(fv)),
        "f0_max": float(np.max(fv)),
        "f0_range": float(np.max(fv) - np.min(fv)),
        "f0_slope": slope,
    }

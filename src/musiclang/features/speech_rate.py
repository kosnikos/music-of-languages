"""Speech-rate estimation via syllable-nuclei detection.

Reference (method & maths): De Jong & Wempe (2009), "Praat script to detect syllable
nuclei and measure speech rate automatically", Behavior Research Methods 41:385-390,
https://www.fon.hum.uva.nl/archive/2009/2009-brm-JongWempe.pdf
"""

from __future__ import annotations

import math

import numpy as np
import parselmouth
from scipy.signal import find_peaks

from musiclang.config import TARGET_SAMPLE_RATE


def count_nuclei(
    intensity_db: np.ndarray, voiced: np.ndarray, min_dip_db: float = 2.0
) -> int:
    """Count voiced intensity peaks separated from their neighbours by a dip of
    at least `min_dip_db` (De Jong & Wempe 2009): adjacent peaks whose between-peak
    valley is less than `min_dip_db` below the lower peak are one nucleus, not two."""
    intensity_db = np.asarray(intensity_db, dtype=float)
    voiced = np.asarray(voiced, dtype=bool)
    peaks, _ = find_peaks(intensity_db)  # all local maxima
    if peaks.size == 0:
        return 0
    retained = [int(peaks[0])]
    for p in peaks[1:]:
        p = int(p)
        q = retained[-1]
        between = intensity_db[q + 1:p]
        valley = float(between.min()) if between.size else min(intensity_db[q], intensity_db[p])
        dip = min(intensity_db[q], intensity_db[p]) - valley
        if dip >= min_dip_db:
            retained.append(p)  # genuinely separate nucleus
        elif intensity_db[p] > intensity_db[q]:
            retained[-1] = p    # same nucleus; keep the higher peak as representative
    return int(sum(1 for p in retained if voiced[p]))


def speech_rate_features(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE
) -> dict[str, float]:
    """Estimate syllable rate from intensity peaks filtered by voicing."""
    duration = len(signal) / sr
    if duration <= 0:
        return {"syllables_per_sec": math.nan, "n_syllables": 0.0, "duration_s": 0.0}

    sound = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=sr)
    intensity = sound.to_intensity(time_step=0.01)
    pitch = sound.to_pitch(time_step=0.01)

    db = np.array(intensity.values[0])
    times = intensity.xs()
    voiced = np.array(
        [(pitch.get_value_at_time(t) or 0.0) > 0 for t in times], dtype=bool
    )
    n = count_nuclei(db, voiced, min_dip_db=2.0)
    return {
        "syllables_per_sec": float(n / duration),
        "n_syllables": float(n),
        "duration_s": float(duration),
    }

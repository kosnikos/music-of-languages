import numpy as np

from musiclang.features import speech_rate


def test_count_nuclei_three_clear_peaks():
    # Three peaks separated by deep dips, all voiced.
    db = np.array([40, 60, 40, 62, 41, 58, 40], dtype=float)
    voiced = np.ones_like(db, dtype=bool)
    assert speech_rate.count_nuclei(db, voiced, min_dip_db=2.0) == 3


def test_count_nuclei_ignores_unvoiced_peaks():
    db = np.array([40, 60, 40, 62, 40], dtype=float)
    voiced = np.array([1, 0, 1, 1, 1], dtype=bool)  # first peak unvoiced
    assert speech_rate.count_nuclei(db, voiced, min_dip_db=2.0) == 1


def test_count_nuclei_requires_min_dip():
    # Tiny ripples (dip < threshold) should not count as separate nuclei.
    db = np.array([40, 60, 59, 60, 40], dtype=float)
    voiced = np.ones_like(db, dtype=bool)
    assert speech_rate.count_nuclei(db, voiced, min_dip_db=2.0) == 1

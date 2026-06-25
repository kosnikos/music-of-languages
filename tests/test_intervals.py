import pytest
import numpy as np

from musiclang.features import intervals


def test_frames_to_intervals_basic():
    # V C V  pattern: 2 vocalic frames, 1 consonantal, 2 vocalic; step 0.01 s
    mask = [True, True, False, True, True]
    voc, cons = intervals.frames_to_intervals(mask, frame_step=0.01)
    assert voc == pytest.approx([0.02, 0.02])
    assert cons == pytest.approx([0.01])


def test_leading_and_trailing_nonvocalic_are_not_consonantal():
    # Only gaps BETWEEN vocalic runs count as consonantal intervals.
    mask = [False, True, False, False, True, False]
    voc, cons = intervals.frames_to_intervals(mask, frame_step=0.01)
    assert voc == pytest.approx([0.01, 0.01])
    assert cons == pytest.approx([0.02])


def test_all_silence_yields_nothing():
    voc, cons = intervals.frames_to_intervals([False, False], frame_step=0.01)
    assert voc == []
    assert cons == []


def test_detect_intervals_on_synthetic_voiced_tone_runs():
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 150 Hz tone (voiced) with a silent gap in the middle.
    tone = 0.3 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
    tone[sr // 2 - 500: sr // 2 + 500] = 0.0
    voc, cons = intervals.detect_intervals(tone, sr=sr)
    assert sum(voc) > 0.0  # detected some vocalic content

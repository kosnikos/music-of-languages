import numpy as np
from musiclang.clean.select import select_segment, SEGMENT_LENGTH_S


def test_select_returns_first_30s_when_enough_speech():
    sr = 16_000
    signal = np.arange(sr * 45, dtype=np.float32)  # 45 s of clean speech
    seg = select_segment(signal, sr=sr)
    assert seg is not None
    assert len(seg) == int(SEGMENT_LENGTH_S * sr)
    assert seg[0] == 0.0 and seg[-1] == float(int(SEGMENT_LENGTH_S * sr) - 1)  # first window


def test_select_returns_none_when_under_30s():
    sr = 16_000
    signal = np.zeros(sr * 20, dtype=np.float32)  # only 20 s clean
    assert select_segment(signal, sr=sr) is None


def test_select_exactly_30s_is_accepted():
    sr = 16_000
    signal = np.zeros(int(SEGMENT_LENGTH_S * sr), dtype=np.float32)
    seg = select_segment(signal, sr=sr)
    assert seg is not None and len(seg) == int(SEGMENT_LENGTH_S * sr)

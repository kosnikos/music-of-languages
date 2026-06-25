import numpy as np

from musiclang.clean.window import Window, window_signal


def test_exact_multiple_gives_full_windows():
    sr = 16_000
    sig = np.arange(5 * sr, dtype=np.float32)
    wins = window_signal(sig, sr, length_s=1.0)
    assert len(wins) == 5
    assert [round(w.start_s, 3) for w in wins] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(len(w.samples) == sr for w in wins)
    assert isinstance(wins[0], Window)


def test_short_tail_is_dropped_by_default():
    sr = 16_000
    sig = np.zeros(int(5.4 * sr), dtype=np.float32)  # 0.4s tail
    wins = window_signal(sig, sr, length_s=1.0)
    assert len(wins) == 5  # the 0.4s tail < min_s(=1.0) is dropped


def test_full_length_returns_single_window():
    sr = 16_000
    sig = np.zeros(3 * sr, dtype=np.float32)
    wins = window_signal(sig, sr, length_s=None)
    assert len(wins) == 1
    assert len(wins[0].samples) == 3 * sr


def test_empty_signal_returns_empty():
    assert window_signal(np.zeros(0, dtype=np.float32), 16_000, length_s=1.0) == []
    assert window_signal(np.zeros(0, dtype=np.float32), 16_000, length_s=None) == []

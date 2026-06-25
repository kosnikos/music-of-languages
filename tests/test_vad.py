import numpy as np

from musiclang.clean import vad


def test_merge_segments_joins_close_spans():
    segs = [(0.0, 1.0), (1.1, 2.0), (5.0, 6.0)]
    merged = vad.merge_segments(segs, gap=0.2)
    assert merged == [(0.0, 2.0), (5.0, 6.0)]


def test_merge_segments_keeps_distant_spans():
    segs = [(0.0, 1.0), (3.0, 4.0)]
    assert vad.merge_segments(segs, gap=0.2) == [(0.0, 1.0), (3.0, 4.0)]


def test_merge_segments_empty():
    assert vad.merge_segments([], gap=0.2) == []


def test_total_speech_seconds():
    assert vad.total_speech_seconds([(0.0, 1.0), (2.0, 3.5)]) == 2.5


def test_concat_speech_picks_correct_samples():
    sr = 1000
    signal = np.arange(3000, dtype=np.float32)  # 3 s at 1 kHz
    out = vad.concat_speech(signal, [(0.0, 1.0), (2.0, 3.0)], sr=sr)
    expected = np.concatenate([np.arange(0, 1000), np.arange(2000, 3000)]).astype(np.float32)
    assert np.array_equal(out, expected)

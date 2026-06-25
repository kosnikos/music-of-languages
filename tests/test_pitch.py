import numpy as np

from musiclang.features import pitch


def test_pitch_features_constant_tone():
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    feats = pitch.pitch_features(tone, sr=sr)
    assert set(feats) == {
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope",
    }
    # Parselmouth should recover ~200 Hz for a clean 200 Hz tone.
    assert abs(feats["f0_mean"] - 200.0) < 10.0
    assert feats["f0_std"] < 5.0


def test_pitch_features_silence_is_nan_safe():
    feats = pitch.pitch_features(np.zeros(16_000, dtype=np.float32), sr=16_000)
    assert all(isinstance(v, float) for v in feats.values())
    assert np.isnan(feats["f0_mean"])

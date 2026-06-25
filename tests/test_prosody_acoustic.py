import numpy as np

from musiclang.features.base import FeatureExtractor
from musiclang.features.prosody_acoustic import ProsodyAcousticExtractor

EXPECTED_KEYS = {
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope",
    "syllables_per_sec", "n_syllables", "duration_s",
    "percent_v", "delta_v", "delta_c", "varco_v", "varco_c", "npvi_v", "rpvi_c",
}


def test_extractor_is_a_feature_extractor():
    ex = ProsodyAcousticExtractor()
    assert isinstance(ex, FeatureExtractor)
    assert ex.name == "prosody_acoustic"


def test_extract_returns_all_keys_as_floats():
    sr = 16_000
    rng = np.random.default_rng(1)
    # 2 s of amplitude-modulated voiced tone -> exercises pitch/rate/intervals.
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))  # 4 Hz syllable-like modulation
    sig = (env * 0.3 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
    sig += rng.normal(0, 0.001, sig.shape).astype(np.float32)

    feats = ProsodyAcousticExtractor().extract(sig, sr=sr)
    assert set(feats) == EXPECTED_KEYS
    assert all(isinstance(v, float) for v in feats.values())
    assert feats["duration_s"] > 1.5

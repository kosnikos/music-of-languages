import numpy as np
import soundfile as sf

from musiclang import audio


def test_normalize_loudness_sets_target_rms():
    sig = np.random.default_rng(0).normal(0, 0.5, 16_000).astype(np.float32)
    out = audio.normalize_loudness(sig, target_rms=0.1)
    rms = float(np.sqrt(np.mean(out**2)))
    assert abs(rms - 0.1) < 1e-3


def test_normalize_loudness_silent_signal_is_safe():
    sig = np.zeros(1000, dtype=np.float32)
    out = audio.normalize_loudness(sig, target_rms=0.1)
    assert np.allclose(out, 0.0)
    assert not np.any(np.isnan(out))


def test_load_audio_returns_mono_target_sr(tmp_path):
    # Write a 2-channel 8 kHz wav; expect mono 16 kHz back.
    sr_in = 8_000
    t = np.linspace(0, 1.0, sr_in, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([tone, tone], axis=1).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(p, stereo, sr_in)

    out = audio.load_audio(p, sr=16_000)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert abs(len(out) - 16_000) <= 2  # resampled to ~1 s at 16 kHz

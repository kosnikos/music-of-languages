import numpy as np
import pytest
import torch

from musiclang.features.base import FeatureExtractor
from musiclang.features import ssl_embedding
from musiclang.features.ssl_embedding import SSLEmbeddingExtractor


class _FakeFeat:
    sampling_rate = 16_000

    def __call__(self, signal, sampling_rate, return_tensors):
        return {"input_values": torch.tensor(np.asarray(signal), dtype=torch.float32).reshape(1, -1)}


class _FakeOut:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeModel:
    """hidden_states[k] is a (1, T, H) tensor filled with value k."""

    def __init__(self, n_layers=4, hidden=6, frames=10):
        self.n_layers, self.hidden, self.frames = n_layers, hidden, frames

    def __call__(self, output_hidden_states, **inputs):
        hs = tuple(
            torch.full((1, self.frames, self.hidden), float(k)) for k in range(self.n_layers)
        )
        return _FakeOut(hs)


def _patch(monkeypatch):
    monkeypatch.setattr(ssl_embedding, "_load_model", lambda model_id, device: (_FakeFeat(), _FakeModel()))


def test_implements_interface_and_name(monkeypatch):
    _patch(monkeypatch)
    ex = SSLEmbeddingExtractor(model_id="facebook/wav2vec2-xls-r-300m", layer=2, pooling="mean")
    assert isinstance(ex, FeatureExtractor)
    assert ex.name == "ssl_wav2vec2-xls-r-300m_l2_mean"


def test_mean_pooling_selects_layer(monkeypatch):
    _patch(monkeypatch)
    ex = SSLEmbeddingExtractor(layer=2, pooling="mean")
    out = ex.extract(np.zeros(16_000, dtype=np.float32), sr=16_000)
    assert len(out) == 6  # hidden size
    assert list(out) == [f"emb_{i:03d}" for i in range(6)]
    assert all(v == 2.0 for v in out.values())  # layer 2 filled with 2.0


def test_mean_std_pooling_doubles_dim(monkeypatch):
    _patch(monkeypatch)
    ex = SSLEmbeddingExtractor(layer=1, pooling="mean_std")
    out = ex.extract(np.zeros(16_000, dtype=np.float32), sr=16_000)
    assert len(out) == 12  # mean(6) + std(6)
    assert all(v == 1.0 for v in list(out.values())[:6])  # means = layer value
    assert all(v == 0.0 for v in list(out.values())[6:])  # std of constant = 0


def test_extract_layers_one_pass_multi_layer(monkeypatch):
    _patch(monkeypatch)  # _FakeModel: 4 layers, hidden=6; hidden_states[k] filled with float k
    ex = SSLEmbeddingExtractor(pooling="mean")
    out = ex.extract_layers(np.zeros(16_000, dtype=np.float32), sr=16_000, layers=(1, 3, -1))
    assert set(out) == {1, 3, -1}
    assert list(out[1]) == [f"emb_{i:03d}" for i in range(6)]
    assert all(v == 1.0 for v in out[1].values())   # layer 1 filled with 1.0
    assert all(v == 3.0 for v in out[3].values())    # layer 3 filled with 3.0
    assert all(v == 3.0 for v in out[-1].values())   # -1 == last == index 3


@pytest.mark.slow
def test_real_xlsr_smoke():
    """Downloads wav2vec2-base (~360MB). Run: uv run pytest -m slow"""
    ex = SSLEmbeddingExtractor(model_id="facebook/wav2vec2-base", layer=-1, pooling="mean")
    rng = np.random.default_rng(0)
    out = ex.extract(rng.standard_normal(16_000).astype(np.float32), sr=16_000)
    assert len(out) == 768
    assert all(np.isfinite(v) for v in out.values())

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "embed_segments", Path(__file__).resolve().parents[1] / "scripts" / "embed_segments.py"
)
embed_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(embed_mod)
embed_segments = embed_mod.embed_segments
_layer_filename = embed_mod._layer_filename


class _FakeLayerExtractor:
    def extract_layers(self, signal, sr, layers):
        # deterministic per-layer vector; length 2
        return {ly: {"emb_000": float(ly), "emb_001": float(len(signal))} for ly in layers}


def test_layer_filename():
    assert _layer_filename(16).endswith("segment_embeddings_xlsr_l16.parquet")
    assert _layer_filename(-1).endswith("segment_embeddings_xlsr_llast.parquet")


def test_embed_segments_writes_one_parquet_per_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "load_audio",
                        lambda path, sr=16_000: np.ones(100, dtype=np.float32))
    manifest = pd.DataFrame([
        {"segment_id": "en_1", "path": "a.wav"},
        {"segment_id": "el_1", "path": "b.wav"},
    ])
    embed_segments(manifest, _FakeLayerExtractor(), layers=(12, 16, -1), out_dir=tmp_path)
    df16 = pd.read_parquet(tmp_path / "segment_embeddings_xlsr_l16.parquet")
    assert list(df16.index) == ["en_1", "el_1"]
    assert df16.loc["en_1", "emb_000"] == 16.0
    assert df16.loc["en_1", "emb_001"] == 100.0
    # resumable: re-running does not duplicate rows
    embed_segments(manifest, _FakeLayerExtractor(), layers=(12, 16, -1), out_dir=tmp_path)
    assert len(pd.read_parquet(tmp_path / "segment_embeddings_xlsr_l16.parquet")) == 2


def test_resumes_only_missing_segment_on_partial_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "load_audio", lambda path, sr=16_000: np.ones(100, dtype=np.float32))

    class _CountingExtractor:
        def __init__(self):
            self.calls = []
        def extract_layers(self, signal, sr, layers):
            self.calls.append(True)
            return {ly: {"emb_000": float(ly), "emb_001": float(len(signal))} for ly in layers}

    manifest = pd.DataFrame([
        {"segment_id": "a", "path": "a.wav"},
        {"segment_id": "b", "path": "b.wav"},
    ])
    # Fully embed "a" across all layers (so it is "done"); leave "b" absent everywhere.
    ex = _CountingExtractor()
    embed_segments(pd.DataFrame([{"segment_id": "a", "path": "a.wav"}]), ex, layers=(12, 16, -1), out_dir=tmp_path)
    ex2 = _CountingExtractor()
    embed_segments(manifest, ex2, layers=(12, 16, -1), out_dir=tmp_path)
    # only "b" should be embedded on the second run
    assert len(ex2.calls) == 1
    for ly_tag in ("l12", "l16", "llast"):
        df = pd.read_parquet(tmp_path / f"segment_embeddings_xlsr_{ly_tag}.parquet")
        assert set(df.index) == {"a", "b"}

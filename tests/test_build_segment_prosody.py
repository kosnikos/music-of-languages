import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "build_segment_prosody", Path(__file__).resolve().parents[1] / "scripts" / "build_segment_prosody.py"
)
bsp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bsp)


class _FakeExtractor:
    @property
    def name(self):
        return "fake"

    def extract(self, signal, sr):
        return {"npvi_v": float(signal.mean()), "varco_v": float(len(signal))}


def test_build_writes_prosody_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(bsp.pipeline, "load_audio", lambda path, sr=16_000: np.ones(16_000, dtype=np.float32))
    manifest = pd.DataFrame([
        {"segment_id": "en_1", "language": "english", "channel_id": "BBC",
         "source": "podcast", "recording_ref": "ep1", "path": "a.wav"},
    ])
    bsp.build(manifest, _FakeExtractor(), out_dir=tmp_path)
    feats = pd.read_parquet(tmp_path / "segment_features_prosody.parquet")
    prov = pd.read_parquet(tmp_path / "segment_provenance.parquet")
    assert feats.loc["en_1", "varco_v"] == 16_000
    assert prov.loc["en_1", "channel_id"] == "BBC"

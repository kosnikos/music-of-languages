# tests/test_pipeline.py
import numpy as np
import pandas as pd

from musiclang import pipeline
from musiclang.pipeline import build_segment_features, segment_clip


class _FakeExtractor:
    @property
    def name(self):
        return "fake"

    def extract(self, signal, sr):
        return {"f0": float(signal.mean()), "n": float(len(signal))}


def test_segment_clip_emits_provenance():
    sig = np.zeros(2 * 16_000, dtype=np.float32)
    segs = segment_clip("english_01", "english", "BBC", sig, 16_000, length_s=1.0)
    assert len(segs) == 2
    meta0, samples0 = segs[0]
    assert meta0["segment_id"] == "english_01_w000"
    assert meta0["clip_id"] == "english_01"
    assert meta0["language"] == "english"
    assert meta0["station_name"] == "BBC"
    assert meta0["window_index"] == 0
    assert len(samples0) == 16_000


def test_build_segment_features(monkeypatch):
    # 2s of constant audio -> 2 one-second windows per clip
    monkeypatch.setattr(pipeline, "clean_clip", lambda path, sr=16_000: np.ones(2 * 16_000, dtype=np.float32))
    manifest = pd.DataFrame(
        [{"clip_id": "english_01", "language": "english", "station_name": "BBC", "path": "x.wav"}]
    )
    seg_df, feat_df = build_segment_features(manifest, _FakeExtractor(), length_s=1.0)
    assert list(seg_df.index) == ["english_01_w000", "english_01_w001"]
    assert seg_df.loc["english_01_w000", "language"] == "english"
    assert list(feat_df.columns) == ["f0", "n"]
    assert feat_df.loc["english_01_w000", "f0"] == 1.0
    assert feat_df.loc["english_01_w000", "n"] == 16_000

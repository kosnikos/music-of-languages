import numpy as np
import pandas as pd

from musiclang import pipeline
from musiclang.pipeline import build_segment_features, build_segment_features_direct, segment_clip


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


def test_build_segment_features_direct_no_reclean(monkeypatch):
    # load_audio is monkeypatched: the loader must NOT call clean_clip / window.
    monkeypatch.setattr(pipeline, "load_audio", lambda path, sr=16_000: np.ones(16_000, dtype=np.float32))
    manifest = pd.DataFrame([{
        "segment_id": "english_seg01", "language": "english", "channel_id": "BBC",
        "source": "podcast", "recording_ref": "ep1", "path": "x.wav",
    }])
    prov_df, feat_df = build_segment_features_direct(manifest, _FakeExtractor())
    assert list(prov_df.index) == ["english_seg01"]
    assert prov_df.loc["english_seg01", "language"] == "english"
    assert prov_df.loc["english_seg01", "channel_id"] == "BBC"
    assert prov_df.loc["english_seg01", "source"] == "podcast"
    assert prov_df.loc["english_seg01", "recording_ref"] == "ep1"
    assert set(feat_df.columns) == {"f0", "n"}
    assert feat_df.loc["english_seg01", "n"] == 16_000  # whole signal, not a 1s window
    assert len(feat_df) == 1  # exactly one row per segment


def test_build_segment_features_direct_empty_manifest():
    manifest = pd.DataFrame(columns=["segment_id", "language", "channel_id", "path"])
    prov_df, feat_df = build_segment_features_direct(manifest, _FakeExtractor())
    assert len(prov_df) == 0
    assert prov_df.index.name == "segment_id"
    assert list(prov_df.columns) == ["language", "channel_id", "source", "recording_ref"]
    assert len(feat_df) == 0

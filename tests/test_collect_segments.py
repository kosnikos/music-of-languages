# tests/test_collect_segments.py
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "collect_segments", Path(__file__).resolve().parents[1] / "scripts" / "collect_segments.py"
)
collect_segments = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect_segments)

from musiclang.verify.verifier import Verdict  # noqa: E402

_META = dict(language="english", source="podcast", channel_id="bbc",
             recording_ref="ep1", recorded_at="2026-06-30T10:00:00Z")


def _verdict(label):
    return Verdict(label, 0.9, "english", "hello world here", 0.9, 0.1, "whisper")


def test_process_recording_keeps_verified_target_speech(tmp_path):
    wav = tmp_path / "r.wav"; wav.write_bytes(b"x")
    status, meta, samples = collect_segments.process_recording(
        wav, **_META,
        clean=lambda p: np.zeros(16_000 * 40, dtype=np.float32),
        select=lambda sig, sr: np.zeros(16_000 * 30, dtype=np.float32),
        verify=lambda seg, sr, lang: _verdict("target-speech"),
    )
    assert status == "kept"
    assert meta["language"] == "english" and meta["label"] == "target-speech"
    assert meta["clean_speech_s"] == 40.0 and len(samples) == 16_000 * 30


def test_process_recording_drops_when_under_30s(tmp_path):
    wav = tmp_path / "r.wav"; wav.write_bytes(b"x")
    status, reason, detail = collect_segments.process_recording(
        wav, **_META,
        clean=lambda p: np.zeros(16_000 * 10, dtype=np.float32),
        select=lambda sig, sr: None,
        verify=lambda seg, sr, lang: _verdict("target-speech"),
    )
    assert status == "drop" and reason == "no-30s"


def test_process_recording_drops_on_verdict(tmp_path):
    wav = tmp_path / "r.wav"; wav.write_bytes(b"x")
    status, reason, detail = collect_segments.process_recording(
        wav, **_META,
        clean=lambda p: np.zeros(16_000 * 40, dtype=np.float32),
        select=lambda sig, sr: np.zeros(16_000 * 30, dtype=np.float32),
        verify=lambda seg, sr, lang: _verdict("music"),
    )
    assert status == "drop" and reason == "music"


def test_process_recording_never_raises(tmp_path):
    wav = tmp_path / "r.wav"; wav.write_bytes(b"x")
    status, reason, detail = collect_segments.process_recording(
        wav, **_META, clean=lambda p: (_ for _ in ()).throw(RuntimeError("bad audio")))
    assert status == "drop" and reason == "error"


def test_run_unknown_kind_is_dropped_not_crash(tmp_path):
    import pandas as pd
    inst = tmp_path / "instances.parquet"
    pd.DataFrame([{"source": "radio", "language": "english", "channel_id": "weird-station",
                   "kind": "weird", "ref": "http://x", "notes": ""}]).to_parquet(inst)
    # An unknown kind short-circuits before any real capture/network/model call.
    seg_df, drop_df = collect_segments.run(
        pilot=True, instances_path=inst, out_dir=tmp_path / "segments")
    assert len(seg_df) == 0
    assert (drop_df["reason"] == "capture-failed").any()
    assert (drop_df["detail"] == "weird").any()

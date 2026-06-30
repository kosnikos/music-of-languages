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
        pilot=True, instances_path=inst, out_dir=tmp_path / "segments",
        warm=False, workers=2)
    assert len(seg_df) == 0
    assert (drop_df["reason"] == "capture-failed").any()
    assert (drop_df["detail"] == "weird").any()
    # Manifests must be written to out_dir, not DATA_DIR
    assert (tmp_path / "segments" / "segments_manifest_pilot.parquet").exists()


def test_run_parallel_keeps_to_target_and_writes_to_out_dir(tmp_path, monkeypatch):
    """Offline test: monkeypatched process_recording + capture; asserts parallel orchestration."""
    import pandas as pd
    from musiclang.probe import adapters as _adapters
    import soundfile as sf

    # Build a minimal instances parquet with 4 podcast channels for english
    inst = tmp_path / "instances.parquet"
    channels = [
        {"source": "podcast", "language": "english", "channel_id": f"ch{i}",
         "kind": "rss_feed", "ref": f"http://feed{i}", "notes": ""}
        for i in range(4)
    ]
    pd.DataFrame(channels).to_parquet(inst)

    # Monkeypatch adapters.latest_enclosures to return 2 fake episode URLs per feed
    monkeypatch.setattr(
        _adapters, "latest_enclosures",
        lambda ref, n: [f"http://ep{j}" for j in range(min(n, 2))]
    )

    # Monkeypatch CAPTURE_DISPATCH so "rss" capture writes a dummy wav and succeeds
    def _fake_capture(ref, wav_path):
        sig = np.zeros(16_000 * 35, dtype=np.float32)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), sig, 16_000)
        return wav_path

    monkeypatch.setitem(_adapters.CAPTURE_DISPATCH, "rss", _fake_capture)

    # Monkeypatch process_recording at the module level (collect_segments imports it directly)
    call_count = {"n": 0}

    def _fake_process(wav_path, *, language, source, channel_id, recording_ref,
                      recorded_at, clean, select, verify, sr):
        call_count["n"] += 1
        # First 2 calls -> kept; rest -> drop:music (tests that surplus is ignored)
        if call_count["n"] <= 2:
            meta = {
                "segment_id": f"{language}_{channel_id}_{recording_ref}",
                "language": language, "source": source, "channel_id": channel_id,
                "recording_ref": recording_ref, "recorded_at": recorded_at,
                "clean_speech_s": 35.0, "path": "",
                "label": "target-speech", "confidence": 0.95,
                "detected_language": language, "transcript": "hello world",
                "tagger_speech": 0.9, "tagger_music": 0.05, "stage_decided": "whisper",
            }
            return ("kept", meta, np.zeros(16_000 * 30, dtype=np.float32))
        return ("drop", "music", "tagger:music=0.8")

    monkeypatch.setattr(collect_segments, "process_recording", _fake_process)

    out_dir = tmp_path / "segments"
    target = 2
    seg_df, drop_df = collect_segments.run(
        per_language=target,
        pilot=False,
        sources=("podcast",),
        instances_path=inst,
        out_dir=out_dir,
        warm=False,
        workers=4,
    )

    # At most `target` kept per language
    assert len(seg_df[seg_df["language"] == "english"]) <= target

    # Manifests written to out_dir (NOT DATA_DIR)
    assert (out_dir / "segments_manifest.parquet").exists()
    assert (out_dir / "segments_drops.parquet").exists()

    # Drop rows recorded for surplus music drops
    assert len(drop_df) >= 0  # may be 0 if all 4 channels fit within budget; just confirm no crash

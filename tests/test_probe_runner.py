# tests/test_probe_runner.py
import importlib.util
from pathlib import Path

import numpy as np

# scripts/ is not a package — load the module by path.
_SPEC = importlib.util.spec_from_file_location(
    "probe_source", Path(__file__).resolve().parents[1] / "scripts" / "probe_source.py"
)
probe_source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe_source)

from musiclang.probe.core import RecordingRef, ProbeResult  # noqa: E402


def test_expand_instances_expands_rss_feeds_and_passes_others():
    rows = [
        {"source": "podcast", "language": "french", "channel_id": "showA",
         "kind": "rss_feed", "ref": "http://feedA"},
        {"source": "radio", "language": "finnish", "channel_id": "yle",
         "kind": "hls", "ref": "http://x.m3u8"},
    ]
    refs = probe_source.expand_instances(
        rows, k=2, enclosure_fn=lambda url, k: ["http://e1.mp3", "http://e2.mp3"]
    )
    kinds = [r.kind for r in refs]
    assert kinds.count("rss") == 2 and "hls" in kinds
    rss = [r for r in refs if r.kind == "rss"]
    assert rss[0].source == "podcast" and rss[0].ref == "http://e1.mp3"
    assert rss[0].channel_id == "showA"


def test_probe_ref_measures_when_capture_succeeds(tmp_path):
    ref = RecordingRef("radio", "greek", "st1", "progressive", "http://s")
    def fake_capture(r, out, **k):
        Path(out).write_bytes(b"x")
        return Path(out)
    res = probe_source.probe_ref(
        ref, tmp_path,
        capture_dispatch={"progressive": fake_capture},
        load=lambda p: np.zeros(16_000 * 40, dtype=np.float32),
        normalize=lambda s: s,
        speech_fn=lambda sig, sr: [(0.0, 33.0)],
    )
    assert res.capturable is True and res.meets_30s is True and res.clean_speech_s == 33.0


def test_probe_ref_reports_capture_failure(tmp_path):
    ref = RecordingRef("radio", "greek", "st1", "hls", "http://s")
    res = probe_source.probe_ref(
        ref, tmp_path, capture_dispatch={"hls": lambda r, o, **k: None}
    )
    assert res.capturable is False and res.error == "capture-failed"


def test_summarize_aggregates_per_source_language():
    results = [
        ProbeResult("radio", "greek", "a", "hls", True, 33.0, True),
        ProbeResult("radio", "greek", "b", "hls", True, 10.0, False),
        ProbeResult("radio", "greek", "a", "hls", False, 0.0, False, "capture-failed"),
    ]
    df = probe_source.summarize(results)
    row = df[(df["source"] == "radio") & (df["language"] == "greek")].iloc[0]
    assert row["n"] == 3
    assert abs(row["capture_rate"] - 2 / 3) < 1e-9
    assert row["distinct_channels"] == 2          # 'a' and 'b'
    assert row["median_clean_s"] == 10.0

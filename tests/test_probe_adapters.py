# tests/test_probe_adapters.py
from pathlib import Path
from types import SimpleNamespace

from musiclang.probe import adapters
from musiclang.probe.core import RecordingRef


def _fake_runner(captured):
    def run(cmd, **kwargs):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFFfake")
        captured.append(cmd)
        return SimpleNamespace(returncode=0)
    return run


def test_capture_hls_builds_streamlink_then_ffmpeg(tmp_path):
    cmds = []
    ref = RecordingRef("radio", "finnish", "yle", "hls", "http://x.m3u8")
    out = tmp_path / "o.wav"
    res = adapters.capture_hls(ref, out, duration_s=60, runner=_fake_runner(cmds))
    assert res == out and out.exists()
    assert cmds[0][0] == "streamlink" and "http://x.m3u8" in cmds[0] and "00:01:00" in cmds[0]
    assert cmds[1][0] == "ffmpeg" and cmds[1][-1] == str(out)


def test_capture_progressive_delegates_to_record_clip(tmp_path):
    cmds = []
    ref = RecordingRef("radio", "spanish", "ser", "progressive", "http://stream")
    out = tmp_path / "o.wav"
    res = adapters.capture_progressive(ref, out, duration_s=45, runner=_fake_runner(cmds))
    assert res == out and out.exists()
    ff = cmds[0]
    assert ff[0] == "ffmpeg" and "http://stream" in ff and "45" in ff and ff[-1] == str(out)


def test_capture_progressive_returns_none_on_failure(tmp_path):
    def boom(cmd, **kwargs):
        raise RuntimeError("ffmpeg failed")
    ref = RecordingRef("radio", "spanish", "ser", "progressive", "http://stream")
    assert adapters.capture_progressive(ref, tmp_path / "o.wav", runner=boom) is None


def test_capture_rss_downloads_then_slices(tmp_path):
    cmds = []
    def fake_dl(url, dest):
        Path(dest).write_bytes(b"media")
        return dest
    ref = RecordingRef("podcast", "french", "show", "rss", "http://ep.mp3")
    out = tmp_path / "o.wav"
    res = adapters.capture_rss(ref, out, downloader=fake_dl, runner=_fake_runner(cmds))
    assert res == out
    ff = cmds[0]
    assert ff[0] == "ffmpeg" and "-ss" in ff and "-t" in ff


def test_capture_rss_returns_none_when_download_fails(tmp_path):
    ref = RecordingRef("podcast", "french", "show", "rss", "http://ep.mp3")
    res = adapters.capture_rss(
        ref, tmp_path / "o.wav", downloader=lambda u, d: None, runner=lambda *a, **k: None
    )
    assert res is None


def test_capture_local_returns_existing_path(tmp_path):
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"x")
    ref = RecordingRef("corpus", "greek", "spk1", "corpus", str(wav))
    assert adapters.capture_local(ref, tmp_path / "ignored.wav") == wav


def test_latest_enclosures_extracts_audio_hrefs():
    feed = SimpleNamespace(entries=[
        SimpleNamespace(enclosures=[{"href": "http://a.mp3", "type": "audio/mpeg"}]),
        SimpleNamespace(enclosures=[{"href": "http://b.jpg", "type": "image/jpeg"}]),
        SimpleNamespace(enclosures=[{"href": "http://c.mp3", "type": "audio/mpeg"}]),
    ])
    urls = adapters.latest_enclosures("http://feed", 2, parser=lambda u: feed)
    assert urls == ["http://a.mp3", "http://c.mp3"]


def test_capture_dispatch_maps_kinds():
    assert set(adapters.CAPTURE_DISPATCH) == {"progressive", "hls", "rss", "corpus"}


# ---------------------------------------------------------------------------
# corpus_probe tests (Task 4)
# ---------------------------------------------------------------------------

def test_corpus_probe_dedups_speakers_and_writes(tmp_path):
    items = [
        {"speaker_id": "s1", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
        {"speaker_id": "s1", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},  # dup
        {"speaker_id": "s2", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
        {"speaker_id": "s3", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
    ]
    written = []

    def fake_writer(path, arr, sr):
        Path(path).write_bytes(b"wav")
        written.append((path, sr))

    pairs = adapters.corpus_probe(
        "finnish", 2, tmp_path, loader=lambda lang: iter(items), writer=fake_writer
    )
    assert len(pairs) == 2
    assert [ref.channel_id for ref, _ in pairs] == ["s1", "s2"]   # deduped, capped at n
    assert all(ref.kind == "corpus" and ref.source == "corpus" for ref, _ in pairs)
    assert all(Path(p).exists() for _, p in pairs)
    assert written[0][1] == 16_000


def test_corpus_probe_greek_uses_client_id(tmp_path):
    """Greek/Common Voice items carry client_id, not speaker_id."""
    items = [
        {"client_id": "c1", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
        {"client_id": "c1", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},  # dup
        {"client_id": "c2", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
    ]
    written = []

    def fake_writer(path, arr, sr):
        Path(path).write_bytes(b"wav")
        written.append(path)

    pairs = adapters.corpus_probe(
        "greek", 3, tmp_path, loader=lambda lang: iter(items), writer=fake_writer
    )
    assert len(pairs) == 2   # only 2 distinct client_ids
    channel_ids = [ref.channel_id for ref, _ in pairs]
    assert channel_ids == ["c1", "c2"]


def test_corpus_spec_covers_all_eight_languages():
    from musiclang.config import SEED_LANGUAGES
    assert set(adapters._CORPUS_SPEC) == set(SEED_LANGUAGES)
    # Greek comes from Common Voice
    assert "common_voice" in adapters._CORPUS_SPEC["greek"][0]
    # All other 7 use VoxPopuli
    for lang, spec in adapters._CORPUS_SPEC.items():
        if lang != "greek":
            assert spec[0] == "facebook/voxpopuli", f"{lang} should use voxpopuli"

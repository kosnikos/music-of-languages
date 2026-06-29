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

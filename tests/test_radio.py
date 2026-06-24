from pathlib import Path

from musiclang.ingest import radio


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        return _FakeResponse(self._payload)


def test_find_stations_parses_payload_and_filters_blank_urls():
    payload = [
        {"name": "Talk One", "url_resolved": "http://a", "codec": "MP3", "bitrate": 128},
        {"name": "No URL", "url_resolved": "", "codec": "MP3", "bitrate": 128},
        {"name": "Talk Two", "url_resolved": "http://b", "codec": "AAC", "bitrate": 64},
    ]
    session = _FakeSession(payload)
    stations = radio.find_stations("german", limit=10, session=session)
    assert [s.name for s in stations] == ["Talk One", "Talk Two"]
    assert stations[0].url == "http://a"
    # language must be passed to the API
    _, params = session.calls[0]
    assert params["language"] == "german"


def test_record_clip_builds_ffmpeg_command(tmp_path):
    captured = {}

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"RIFFfake")
        class R:  # noqa: D401
            returncode = 0
        return R()

    out = tmp_path / "clip.wav"
    result = radio.record_clip("http://a", out, duration_s=30, runner=fake_runner)
    assert result == out
    assert out.exists()
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "30" in cmd            # duration appears
    assert "http://a" in cmd      # input url appears
    assert str(out) == cmd[-1]    # output path is last

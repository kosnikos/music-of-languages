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
        {"name": "Talk One", "url_resolved": "http://a", "codec": "MP3", "bitrate": 128,
         "country": "Germany", "language": "german"},
        {"name": "No URL", "url_resolved": "", "codec": "MP3", "bitrate": 128,
         "country": "Germany", "language": "german"},
        {"name": "Talk Two", "url_resolved": "http://b", "codec": "AAC", "bitrate": 64,
         "country": "Austria", "language": "german"},
    ]
    session = _FakeSession(payload)
    stations = radio.find_stations("german", limit=10, session=session)
    assert [s.name for s in stations] == ["Talk One", "Talk Two"]
    assert stations[0].url == "http://a"
    # country and language must be parsed onto each Station
    assert stations[0].country == "Germany"
    assert stations[0].language == "german"
    assert stations[1].country == "Austria"
    # language must be passed to the API
    _, params = session.calls[0]
    assert params["language"] == "german"


def test_find_stations_parses_country_and_language_fields():
    """Station.country and Station.language are populated from the API payload."""
    payload = [
        {
            "name": "Alpha Radio",
            "url_resolved": "http://alpha",
            "codec": "MP3",
            "bitrate": 128,
            "country": "Greece",
            "language": "greek",
        },
    ]
    session = _FakeSession(payload)
    stations = radio.find_stations("greek", limit=5, session=session)
    assert len(stations) == 1
    assert stations[0].country == "Greece"
    assert stations[0].language == "greek"


def test_find_stations_defaults_country_and_language_when_absent():
    """Rows without country/language fields produce Station with empty string defaults."""
    payload = [
        {"name": "Bare Station", "url_resolved": "http://bare", "codec": "MP3", "bitrate": 64},
    ]
    session = _FakeSession(payload)
    stations = radio.find_stations("french", limit=5, session=session)
    assert stations[0].country == ""
    assert stations[0].language == ""


def test_station_positional_construction_still_works():
    """Station(name, url, codec, bitrate) still works — new fields have defaults."""
    st = radio.Station("My Station", "http://s", "MP3", 128)
    assert st.name == "My Station"
    assert st.country == ""
    assert st.language == ""


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


# ---------------------------------------------------------------------------
# New tests: geo params + tag in find_stations, find_capital_stations
# ---------------------------------------------------------------------------

def _station_dict(name, url, codec="MP3", bitrate=128, country="", language=""):
    return {
        "name": name,
        "url_resolved": url,
        "codec": codec,
        "bitrate": bitrate,
        "country": country,
        "language": language,
    }


class _GeoAwareFakeSession:
    """Fake session that branches on whether geo_lat is in params.

    - Geo queries (geo_lat present): returns a small list (controlled per tag).
    - Nationwide queries (no geo_lat): returns a larger fixed list.
    """

    def __init__(self, geo_stations_by_tag, nationwide_stations):
        # geo_stations_by_tag: dict[tag_str, list[dict]]  — if tag absent, returns []
        self._geo = geo_stations_by_tag
        self._nationwide = nationwide_stations
        self.calls = []  # list of params dicts

    def get(self, url, params=None, timeout=None, headers=None):
        params = params or {}
        self.calls.append(dict(params))
        if "geo_lat" in params:
            tag = params.get("tag", "")
            payload = self._geo.get(tag, [])
        else:
            payload = self._nationwide
        return _FakeResponse(payload)


def test_find_stations_includes_geo_and_tag_params_when_given():
    """find_stations passes geo_lat/geo_long/geo_distance/tag to the API."""
    session = _FakeSession([])
    radio.find_stations(
        "english",
        limit=5,
        tag="talk",
        geo_lat=51.5074,
        geo_long=-0.1278,
        geo_distance=60_000,
        session=session,
    )
    _, params = session.calls[0]
    assert params["tag"] == "talk"
    assert params["geo_lat"] == 51.5074
    assert params["geo_long"] == -0.1278
    assert params["geo_distance"] == 60_000
    # tagList should NOT appear when tag is provided without explicit tags arg
    assert "tagList" not in params


def test_find_stations_omits_geo_and_tag_when_not_given():
    """find_stations does not include geo or tag keys when they are not supplied."""
    session = _FakeSession([])
    radio.find_stations("german", limit=5, session=session)
    _, params = session.calls[0]
    assert "geo_lat" not in params
    assert "geo_long" not in params
    assert "geo_distance" not in params
    assert "tag" not in params


def test_find_capital_stations_unions_speech_tags_and_dedups_by_url():
    """find_capital_stations deduplicates stations returned by multiple tag calls."""
    # Station A appears under both "talk" and "news" — should appear once in result.
    shared = _station_dict("Shared", "http://shared", country="United Kingdom", language="english")
    talk_only = _station_dict("TalkOnly", "http://talk-only", country="United Kingdom", language="english")
    news_only = _station_dict("NewsOnly", "http://news-only", country="United Kingdom", language="english")

    geo_by_tag = {
        "talk": [shared, talk_only],
        "news": [shared, news_only],
        # other tags return nothing
    }
    # Nationwide would return different stations — should NOT be called if we
    # have enough capital stations.
    nationwide = [_station_dict("Nationwide", "http://nationwide")]

    session = _GeoAwareFakeSession(geo_by_tag, nationwide)
    stations = radio.find_capital_stations("english", limit=10, session=session, min_capital=3)

    urls = [s.url for s in stations]
    # Dedup: shared appears once
    assert urls.count("http://shared") == 1
    # Both unique stations present
    assert "http://talk-only" in urls
    assert "http://news-only" in urls
    # We have 3 unique capital stations (≥ min_capital=3), so nationwide NOT used
    assert "http://nationwide" not in urls


def test_find_capital_stations_falls_back_to_nationwide_when_too_few():
    """When capital geo queries return fewer than min_capital stations, nationwide is appended."""
    # Only 1 unique geo station (below min_capital=3)
    geo_by_tag = {
        "talk": [_station_dict("GeoTalk", "http://geo-talk")],
        # all other tags return nothing
    }
    nationwide = [
        _station_dict("Nation1", "http://nation1"),
        _station_dict("Nation2", "http://nation2"),
        _station_dict("Nation3", "http://nation3"),
    ]

    session = _GeoAwareFakeSession(geo_by_tag, nationwide)
    stations = radio.find_capital_stations("english", limit=10, session=session, min_capital=3)

    urls = [s.url for s in stations]
    # Capital station preserved
    assert "http://geo-talk" in urls
    # Nationwide stations appended
    assert "http://nation1" in urls
    assert "http://nation2" in urls


def test_find_capital_stations_enough_capital_does_not_trigger_fallback():
    """When capital geo queries return >= min_capital stations, nationwide is NOT queried."""
    geo_by_tag = {
        "talk": [
            _station_dict("G1", "http://g1"),
            _station_dict("G2", "http://g2"),
            _station_dict("G3", "http://g3"),
        ],
    }
    nationwide = [_station_dict("Nation", "http://nation")]

    session = _GeoAwareFakeSession(geo_by_tag, nationwide)
    stations = radio.find_capital_stations("english", limit=10, session=session, min_capital=3)

    # Check no nationwide call was made (no call without geo_lat)
    geo_calls = [p for p in session.calls if "geo_lat" in p]
    nation_calls = [p for p in session.calls if "geo_lat" not in p]
    assert len(geo_calls) > 0
    assert len(nation_calls) == 0

    urls = [s.url for s in stations]
    assert "http://nation" not in urls

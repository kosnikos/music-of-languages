"""Lightweight, portable radio ingest helper (Phase 0 prototype).

Phase 1 hardens this (multi-station/time sampling, retries, scaling).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

from musiclang.config import (
    CAPITAL_GEO_DISTANCE_M,
    CAPITALS,
    MIN_CAPITAL_STATIONS,
    SPEECH_TAGS,
)

# A public radio-browser.info mirror. Phase 1 should resolve a mirror dynamically.
API_BASE = "https://de1.api.radio-browser.info"
USER_AGENT = "music-of-languages/0.0 (research; contact: project owner)"


@dataclass(frozen=True)
class Station:
    name: str
    url: str
    codec: str
    bitrate: int
    country: str = ""
    language: str = ""


def find_stations(
    language: str,
    limit: int = 10,
    tags: str | None = "talk,news",
    session=None,
    *,
    tag: str | None = None,
    geo_lat: float | None = None,
    geo_long: float | None = None,
    geo_distance: int | None = None,
) -> list[Station]:
    """Return up to `limit` stations for `language`, biased to talk/news tags.

    Optional geo params (geo_lat, geo_long, geo_distance) narrow results to a
    geographic radius.  ``tag`` sends a single tag via the radio-browser ``tag``
    field (OR-able by repeated calls); ``tags`` sends ``tagList`` (AND semantics).
    When ``tag`` is given without an explicit ``tags``, ``tagList`` is omitted.
    """
    session = session or requests.Session()
    params: dict[str, object] = {
        "language": language,
        "limit": limit,
        "hidebroken": "true",
        "order": "votes",
        "reverse": "true",
    }
    # Include tagList only when tags was explicitly provided (non-None).
    if tags is not None and tag is None:
        params["tagList"] = tags
    # Single-tag field (OR semantics when called repeatedly).
    if tag is not None:
        params["tag"] = tag
    # Geo params — include only when provided.
    if geo_lat is not None:
        params["geo_lat"] = geo_lat
    if geo_long is not None:
        params["geo_long"] = geo_long
    if geo_distance is not None:
        params["geo_distance"] = geo_distance

    resp = session.get(
        f"{API_BASE}/json/stations/search",
        params=params,
        timeout=20,
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    stations: list[Station] = []
    for row in resp.json():
        url = row.get("url_resolved") or row.get("url") or ""
        if not url:
            continue
        stations.append(
            Station(
                name=row.get("name", "").strip(),
                url=url,
                codec=row.get("codec", ""),
                bitrate=int(row.get("bitrate", 0) or 0),
                country=row.get("country", "").strip(),
                language=row.get("language", "").strip(),
            )
        )
    return stations


def find_capital_stations(
    language: str,
    limit: int = 10,
    session=None,
    min_capital: int = MIN_CAPITAL_STATIONS,
) -> list[Station]:
    """Return up to `limit` speech stations near the capital city for `language`.

    Strategy:
    1. For each tag in SPEECH_TAGS, query the radio-browser API with geo params
       centred on the capital.  Deduplicate by URL, preserving first-seen order.
    2. If fewer than `min_capital` unique capital stations are found, append
       nationwide stations (using the standard talk,news tagList) and dedup again.
    3. Return the deduped list truncated to `limit`.

    A single requests.Session is reused across all per-tag calls to avoid opening
    a new TCP connection per tag.
    """
    # Reuse one session for all sub-calls.
    if session is None:
        session = requests.Session()

    cap = CAPITALS.get(language)

    seen_urls: dict[str, Station] = {}  # url -> Station, preserves insertion order

    if cap is not None:
        for speech_tag in SPEECH_TAGS:
            results = find_stations(
                language,
                limit=limit,
                tags=None,
                tag=speech_tag,
                geo_lat=cap.lat,
                geo_long=cap.lon,
                geo_distance=CAPITAL_GEO_DISTANCE_M,
                session=session,
            )
            for st in results:
                if st.url not in seen_urls:
                    seen_urls[st.url] = st

    # Fallback: if too few capital stations, append nationwide results.
    if len(seen_urls) < min_capital:
        nationwide = find_stations(language, limit=limit, tags="talk,news", session=session)
        for st in nationwide:
            if st.url not in seen_urls:
                seen_urls[st.url] = st

    return list(seen_urls.values())[:limit]


def record_clip(
    stream_url: str,
    out_path: str | Path,
    duration_s: int = 60,
    runner=subprocess.run,
) -> Path:
    """Capture `duration_s` seconds of `stream_url` to `out_path` (wav) via ffmpeg."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(duration_s),
        "-i", stream_url,
        "-ac", "1", "-ar", "16000",
        str(out_path),
    ]
    runner(cmd, check=True)
    return out_path

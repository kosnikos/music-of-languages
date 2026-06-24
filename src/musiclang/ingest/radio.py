"""Lightweight, portable radio ingest helper (Phase 0 prototype).

Phase 1 hardens this (multi-station/time sampling, retries, scaling).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

# A public radio-browser.info mirror. Phase 1 should resolve a mirror dynamically.
API_BASE = "https://de1.api.radio-browser.info"
USER_AGENT = "music-of-languages/0.0 (research; contact: project owner)"


@dataclass(frozen=True)
class Station:
    name: str
    url: str
    codec: str
    bitrate: int


def find_stations(
    language: str, limit: int = 10, tags: str = "talk,news", session=None
) -> list[Station]:
    """Return up to `limit` stations for `language`, biased to talk/news tags."""
    session = session or requests.Session()
    params = {
        "language": language,
        "tagList": tags,
        "limit": limit,
        "hidebroken": "true",
        "order": "votes",
        "reverse": "true",
    }
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
            )
        )
    return stations


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

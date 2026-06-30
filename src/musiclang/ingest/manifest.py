"""Per-clip provenance manifest: language, station, time, path.

Provenance (sub-clip -> recording -> station -> language) lets aggregation
estimate dispersion at the recording/station level rather than over correlated
sub-clips, avoiding pseudoreplication of within-recording variation.
"""

from __future__ import annotations

import pandas as pd

MANIFEST_COLUMNS = [
    "clip_id", "language", "station_name", "station_url",
    "country", "recorded_at", "duration_s", "path",
]


def manifest_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build the clip manifest DataFrame with the canonical column order."""
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


SEGMENTS_COLUMNS = [
    "segment_id", "language", "source", "channel_id", "recording_ref", "recorded_at",
    "clean_speech_s", "path",
    "label", "confidence", "detected_language", "transcript",
    "tagger_speech", "tagger_music", "stage_decided",
]


def segments_manifest_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build the verified-segments manifest with the canonical column order."""
    return pd.DataFrame(rows, columns=SEGMENTS_COLUMNS)


DROPS_COLUMNS = ["language", "source", "channel_id", "recording_ref", "reason", "detail"]


def drops_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build the per-language drop log (why each candidate was rejected)."""
    return pd.DataFrame(rows, columns=DROPS_COLUMNS)

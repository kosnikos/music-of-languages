"""Recorded clips -> cleaned speech -> windowed segments -> per-segment features.

One pipeline serves every FeatureExtractor (scalar prosody or SSL embedding):
the per-segment feature table is extractor-agnostic; only the downstream
aggregation/distance differs (scalar: aggregate+standardize+euclidean; embedding:
centroid+cosine, see proximity/embedding.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from musiclang.audio import load_audio, normalize_loudness
from musiclang.clean.vad import concat_speech, extract_speech
from musiclang.clean.window import window_signal
from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.features.base import FeatureExtractor


def clean_clip(path: str | Path, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load -> loudness-normalize -> VAD -> concatenate speech into one array."""
    signal = normalize_loudness(load_audio(path, sr=sr))
    segments = extract_speech(signal, sr=sr)
    return concat_speech(signal, segments, sr=sr)


def segment_clip(
    clip_id: str,
    language: str,
    station_name: str,
    signal: np.ndarray,
    sr: int,
    length_s: float | None,
) -> list[tuple[dict, np.ndarray]]:
    """Window a cleaned clip into (provenance-meta, samples) pairs."""
    out: list[tuple[dict, np.ndarray]] = []
    for i, w in enumerate(window_signal(signal, sr, length_s)):
        meta = {
            "segment_id": f"{clip_id}_w{i:03d}",
            "clip_id": clip_id,
            "language": language,
            "station_name": station_name,
            "window_index": i,
            "start_s": w.start_s,
            "length_s": len(w.samples) / sr,
        }
        out.append((meta, w.samples))
    return out


def build_segment_features(
    manifest: pd.DataFrame,
    extractor: FeatureExtractor,
    length_s: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (segments provenance, per-segment features) for every clip in `manifest`.

    Empty cleaned clips are skipped, but a per-clip failure (e.g. a corrupt clip
    that raises in `clean_clip`/`extract`) propagates: a caller that must tolerate
    partial failure should isolate clips (call this per single-row manifest in a
    try/except), as the comparison notebook does.
    """
    seg_rows: list[dict] = []
    feat_rows: dict[str, dict] = {}
    for _, row in manifest.iterrows():
        signal = clean_clip(row["path"])
        if len(signal) == 0:
            continue
        for meta, samples in segment_clip(
            row["clip_id"], row["language"], row["station_name"],
            signal, TARGET_SAMPLE_RATE, length_s,
        ):
            seg_rows.append(meta)
            feat_rows[meta["segment_id"]] = extractor.extract(samples, sr=TARGET_SAMPLE_RATE)
    seg_df = pd.DataFrame(seg_rows).set_index("segment_id")
    feat_df = pd.DataFrame.from_dict(feat_rows, orient="index").sort_index()
    return seg_df, feat_df

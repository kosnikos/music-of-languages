"""Aggregate per-clip feature vectors into per-language rows (mean + dispersion)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_language(clip_vectors: list[dict[str, float]]) -> dict[str, float]:
    """Per-feature mean and std across clips, ignoring NaNs."""
    keys = sorted({k for v in clip_vectors for k in v})
    out: dict[str, float] = {}
    for k in keys:
        values = np.array([v.get(k, np.nan) for v in clip_vectors], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            out[f"{k}_mean"] = np.nan
            out[f"{k}_std"] = np.nan
        else:
            out[f"{k}_mean"] = float(np.mean(values))
            out[f"{k}_std"] = float(np.std(values))
    return out


def aggregate_language_robust(clip_vectors: list[dict[str, float]]) -> dict[str, float]:
    """Per-feature median + MAD + IQR across clips, ignoring NaNs (robust vs aggregate_language)."""
    keys = sorted({k for v in clip_vectors for k in v})
    out: dict[str, float] = {}
    for k in keys:
        values = np.array([v.get(k, np.nan) for v in clip_vectors], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            out[f"{k}_median"] = out[f"{k}_mad"] = out[f"{k}_iqr"] = np.nan
        else:
            med = float(np.median(values))
            out[f"{k}_median"] = med
            out[f"{k}_mad"] = float(np.median(np.abs(values - med)))
            out[f"{k}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    return out


def build_language_table(
    per_language: dict[str, list[dict[str, float]]]
) -> pd.DataFrame:
    """Build a language-indexed DataFrame of aggregated features."""
    rows = {lang: aggregate_language(vecs) for lang, vecs in per_language.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()

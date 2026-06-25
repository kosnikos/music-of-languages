"""Method-agnostic proximity-space agreement metrics.

These operate on a distance matrix + class labels, so they evaluate ANY feature
method (scalar rhythm features or SSL embeddings) on the same footing.

- silhouette: Rousseeuw (1987), J. Comput. Appl. Math. 20:53-65,
  https://doi.org/10.1016/0377-0427(87)90125-7
- within/between class separation is the standard cluster-validity contrast.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def class_silhouette(dist_df: pd.DataFrame, labels: dict[str, str]) -> float:
    """Silhouette of `labels` using the precomputed distance matrix (higher=better)."""
    keys = [k for k in dist_df.index if k in labels]
    y = [labels[k] for k in keys]
    if len(set(y)) < 2 or len(keys) < 3:
        return math.nan
    sub = dist_df.loc[keys, keys].to_numpy(dtype=float)
    try:
        return float(silhouette_score(sub, y, metric="precomputed"))
    except ValueError:
        return math.nan


def within_between_separation(dist_df: pd.DataFrame, labels: dict[str, str]) -> dict[str, float]:
    """Mean within-class vs between-class distance, their gap and ratio."""
    keys = [k for k in dist_df.index if k in labels]
    within: list[float] = []
    between: list[float] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            d = float(dist_df.loc[a, b])
            (within if labels[a] == labels[b] else between).append(d)
    wm = float(np.mean(within)) if within else math.nan
    bm = float(np.mean(between)) if between else math.nan
    gap = bm - wm
    ratio = wm / bm if bm not in (0.0, math.nan) and not math.isnan(bm) else math.nan
    return {"within_mean": wm, "between_mean": bm, "gap": gap, "ratio": ratio}


def confound_report(
    dist_df: pd.DataFrame,
    language_labels: dict[str, str],
    station_labels: dict[str, str],
) -> dict[str, float]:
    """Does the geometry cluster by language or by station/channel?

    If `station_*` separation rivals/exceeds `language_*`, the method may be
    encoding channel rather than the language's sound (cycle spec §3.6).
    """
    return {
        "language_silhouette": class_silhouette(dist_df, language_labels),
        "station_silhouette": class_silhouette(dist_df, station_labels),
        "language_gap": within_between_separation(dist_df, language_labels)["gap"],
        "station_gap": within_between_separation(dist_df, station_labels)["gap"],
    }

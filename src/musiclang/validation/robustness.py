"""Robust per-language proximity + stability (leave-one-station-out, bootstrap CIs).

proximity_pipeline rebuilds the per-language geometry from ANY segment subset, so
resampling stability just calls it on held-out / resampled segment sets.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from musiclang.features.aggregate import aggregate_language_robust
from musiclang.proximity.distance import distance_matrix, standardize
from musiclang.proximity.embedding import language_centroids


def proximity_pipeline(
    feat_df: pd.DataFrame,
    prov_df: pd.DataFrame,
    method: str,
    exclude: Iterable[str] | None = None,
    weighting: str = "channel",
) -> pd.DataFrame:
    if method not in ("prosody", "ssl"):
        raise ValueError(f"method must be 'prosody' or 'ssl', got {method!r}")
    excl = set(exclude) if exclude is not None else None
    keep = [i for i in feat_df.index if excl is None or i not in excl]
    feats, prov = feat_df.loc[keep], prov_df.loc[keep]
    if method == "prosody":
        table = {}
        for lang, grp in prov.groupby("language"):
            vecs = [feats.loc[sid].to_dict() for sid in grp.index]
            table[lang] = aggregate_language_robust(vecs)
        tbl = pd.DataFrame.from_dict(table, orient="index").sort_index()
        med = tbl[[c for c in tbl.columns if c.endswith("_median")]]
        return distance_matrix(standardize(med), metric="euclidean")
    emb = feats.join(prov[["language", "channel_id"]])
    emb["clip_id"] = emb.index
    cent = language_centroids(emb, weighting=weighting)
    return distance_matrix(cent, metric="cosine")


def leave_one_station_out(feat_df, prov_df, metric_fn: Callable, method, weighting="channel") -> pd.DataFrame:
    rows = []
    for chan, grp in prov_df.groupby("channel_id"):
        dist = proximity_pipeline(feat_df, prov_df, method, exclude=set(grp.index), weighting=weighting)
        rows.append({"channel_id": chan, "n_dropped": len(grp), "metric": float(metric_fn(dist))})
    return pd.DataFrame(rows)


def leave_one_segment_out(feat_df, prov_df, metric_fn: Callable, method, weighting="channel") -> pd.DataFrame:
    rows = []
    for sid in feat_df.index:
        dist = proximity_pipeline(feat_df, prov_df, method, exclude={sid}, weighting=weighting)
        rows.append({"segment_id": sid, "metric": float(metric_fn(dist))})
    return pd.DataFrame(rows)


def bootstrap_metric_ci(
    feat_df, prov_df, metric_fn: Callable, method,
    n_boot: int = 1000, seed: int = 0, weighting: str = "channel", ci: float = 95,
) -> dict:
    rng = np.random.default_rng(seed)
    by_lang = {lang: list(grp.index) for lang, grp in prov_df.groupby("language")}
    vals: list[float] = []
    for _ in range(n_boot):
        picks: list[str] = []
        for ids in by_lang.values():
            picks += rng.choice(ids, size=len(ids), replace=True).tolist()
        new = [f"b{j}" for j in range(len(picks))]
        f = feat_df.loc[picks].copy(); f.index = new
        p = prov_df.loc[picks].copy(); p.index = new
        try:
            v = float(metric_fn(proximity_pipeline(f, p, method, weighting=weighting)))
        except Exception:
            continue
        if not np.isnan(v):
            vals.append(v)
    arr = np.array(vals, dtype=float)
    lo = float(np.percentile(arr, (100 - ci) / 2))
    hi = float(np.percentile(arr, 100 - (100 - ci) / 2))
    return {"point": float(np.median(arr)), "lo": lo, "hi": hi, "n": int(arr.size)}

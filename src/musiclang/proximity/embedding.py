"""Per-language centroid aggregation for embedding features.

Embeddings do NOT use the scalar mean+std -> z-score -> Euclidean path; they use
the mean of L2-normalized clip embeddings, compared with cosine distance. This
mirrors how SSL/x-vector language embeddings are pooled and compared (cosine on
mean-pooled hidden states): wav2vec 2.0 (Baevski et al. 2020, arXiv:2006.11477).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def language_centroids(
    emb_df: pd.DataFrame,
    group: str = "language",
    recording_col: str = "clip_id",
    weighting: str = "recording",
) -> pd.DataFrame:
    """L2-normalize each segment embedding, then average into per-`group` centroids."""
    if weighting not in ("recording", "flat"):
        raise ValueError(f"weighting must be 'recording' or 'flat', got {weighting!r}")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    x = emb_df[emb_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb_df.copy()
    unit[emb_cols] = x / norms
    if weighting == "flat":
        cent = unit.groupby(group)[emb_cols].mean()
    else:
        per_rec = unit.groupby([group, recording_col])[emb_cols].mean()
        cent = per_rec.groupby(level=0).mean()
    return cent.sort_index()

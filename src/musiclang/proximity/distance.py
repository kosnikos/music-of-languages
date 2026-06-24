"""Language proximity: standardization, distances, clustering, MDS embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each numeric column; drop columns that are all-NaN or constant."""
    numeric = df.select_dtypes("number").dropna(axis=1, how="any")
    std = numeric.std(axis=0, ddof=1)
    numeric = numeric.loc[:, std > 0]
    return (numeric - numeric.mean(axis=0)) / numeric.std(axis=0, ddof=1)


def distance_matrix(df: pd.DataFrame, metric: str = "euclidean") -> pd.DataFrame:
    """Square, symmetric language×language distance matrix."""
    condensed = pdist(df.values, metric=metric)
    square = squareform(condensed)
    return pd.DataFrame(square, index=df.index, columns=df.index)


def linkage_matrix(dist_df: pd.DataFrame, method: str = "ward") -> np.ndarray:
    """SciPy linkage matrix for dendrograms, from a square distance frame."""
    condensed = squareform(dist_df.values, checks=False)
    return linkage(condensed, method=method)


def mds_2d(dist_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """2-D metric MDS embedding from a precomputed distance matrix."""
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, normalized_stress=False)
    coords = mds.fit_transform(dist_df.values)
    return pd.DataFrame(coords, index=dist_df.index, columns=["mds_x", "mds_y"])

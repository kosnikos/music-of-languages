"""Swappable per-language outlier detection (mirrors features/base.FeatureExtractor).

Primary method: robust distance-from-centroid (median + MAD z on a 1-D distance),
which stays stable in 1024-dim embeddings at small n where covariance methods fail.
MAD: Leys et al. 2013, https://doi.org/10.1016/j.jesp.2013.03.013
Isolation Forest: Liu et al. 2008, https://doi.org/10.1109/ICDM.2008.17
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

_MAD_TO_SIGMA = 0.6744897501960817  # Phi^-1(0.75): scales MAD to a std-equivalent


@dataclass(frozen=True)
class OutlierResult:
    scores: np.ndarray       # per-segment robust-z (or anomaly score)
    is_outlier: np.ndarray   # bool mask, same row order as X
    threshold: float


class OutlierDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def flag(self, X: np.ndarray) -> OutlierResult:
        """X = (n_segments, n_features) for ONE language."""


class CentroidMADDetector(OutlierDetector):
    def __init__(self, threshold: float = 3.5, metric: str = "euclidean") -> None:
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")
        self.threshold = threshold
        self.metric = metric

    @property
    def name(self) -> str:
        return f"centroid_mad_{self.metric}"

    def flag(self, X: np.ndarray) -> OutlierResult:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit = X / norms
            centroid = unit.mean(axis=0)
            cnorm = np.linalg.norm(centroid) or 1.0
            centroid = centroid / cnorm
            dist = 1.0 - unit @ centroid
        else:
            centroid = X.mean(axis=0)
            dist = np.linalg.norm(X - centroid, axis=1)
        med = float(np.median(dist))
        mad = float(np.median(np.abs(dist - med)))
        if mad == 0.0:
            return OutlierResult(np.zeros(n), np.zeros(n, dtype=bool), self.threshold)
        z = _MAD_TO_SIGMA * (dist - med) / mad
        return OutlierResult(z, z > self.threshold, self.threshold)  # upper tail: far from centroid

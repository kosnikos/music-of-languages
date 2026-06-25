"""The pluggable feature-extraction interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

FeatureVector = dict[str, float]


class FeatureExtractor(ABC):
    """Maps a single speech clip to a flat, named feature vector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used to namespace this extractor's outputs."""

    @abstractmethod
    def extract(self, signal: np.ndarray, sr: int) -> FeatureVector:
        """Return a {feature_name: value} dict for one clip."""


class ConstantExtractor(FeatureExtractor):
    """Trivial extractor used to validate the interface."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    @property
    def name(self) -> str:
        return "constant"

    def extract(self, signal: np.ndarray, sr: int) -> FeatureVector:
        return {"constant": float(self._value)}

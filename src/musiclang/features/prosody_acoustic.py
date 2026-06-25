"""The alignment-free prosody/rhythm FeatureExtractor (Phase 0 primary candidate).

Inspired by the DisVoice prosody feature set, but self-contained (parselmouth +
our own rhythm metrics) so features stay tweakable during exploration.

Per-feature maths/references live in the source modules each feature comes from:
pitch.py (F0/intonation), speech_rate.py (tempo), and intervals.py + rhythm_metrics.py
(%V, ΔC/ΔV, Varcos, nPVI/rPVI). DisVoice reference: https://github.com/jcvasquezc/DisVoice
"""

from __future__ import annotations

import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.features import intervals, pitch, rhythm_metrics, speech_rate
from musiclang.features.base import FeatureExtractor, FeatureVector


class ProsodyAcousticExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "prosody_acoustic"

    def extract(self, signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> FeatureVector:
        feats: FeatureVector = {}
        feats.update(pitch.pitch_features(signal, sr=sr))
        feats.update(speech_rate.speech_rate_features(signal, sr=sr))

        voc, cons = intervals.detect_intervals(signal, sr=sr)
        feats["percent_v"] = rhythm_metrics.percent_v(voc, cons)
        feats["delta_v"] = rhythm_metrics.delta(voc)
        feats["delta_c"] = rhythm_metrics.delta(cons)
        feats["varco_v"] = rhythm_metrics.varco(voc)
        feats["varco_c"] = rhythm_metrics.varco(cons)
        feats["npvi_v"] = rhythm_metrics.npvi(voc)
        feats["rpvi_c"] = rhythm_metrics.rpvi(cons)
        return {k: float(v) for k, v in feats.items()}

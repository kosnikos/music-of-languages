"""Self-supervised speech-embedding feature extractor (wav2vec2 / XLS-R / HuBERT).

Mean-pools (optionally mean+std) a chosen transformer hidden layer into a fixed
vector per clip. Distance between languages is cosine on the per-language
centroid (see proximity/embedding.py). Pooling/distance choice and models:

- wav2vec 2.0: Baevski et al. (2020), arXiv:2006.11477
- XLS-R (multilingual): Babu et al. (2021), arXiv:2111.09296
- HuBERT: Hsu et al. (2021), arXiv:2106.07447

Mid layers tend to carry the most linguistic/phonetic information, so `layer`
is configurable and swept rather than fixed (see the cycle spec, §3.1).
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.features.base import FeatureExtractor, FeatureVector


@lru_cache(maxsize=2)
def _load_model(model_id: str, device: str):
    """Load and cache (feature_extractor, model). Monkeypatched in unit tests."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    feat = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return feat, model


class SSLEmbeddingExtractor(FeatureExtractor):
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xls-r-300m",
        layer: int = -1,
        pooling: str = "mean",
        device: str = "cpu",
    ) -> None:
        if pooling not in ("mean", "mean_std"):
            raise ValueError(f"pooling must be 'mean' or 'mean_std', got {pooling!r}")
        self.model_id = model_id
        self.layer = layer
        self.pooling = pooling
        self.device = device

    @property
    def name(self) -> str:
        short = self.model_id.split("/")[-1]
        return f"ssl_{short}_l{self.layer}_{self.pooling}"

    def extract(self, signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> FeatureVector:
        import torch

        feat, model = _load_model(self.model_id, self.device)
        model_sr = int(getattr(feat, "sampling_rate", TARGET_SAMPLE_RATE))
        if sr != model_sr:
            signal = librosa.resample(signal.astype(np.float32), orig_sr=sr, target_sr=model_sr)
        inputs = feat(signal, sampling_rate=model_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[self.layer].squeeze(0)  # (T, H)
        if self.pooling == "mean":
            pooled = hidden.mean(dim=0)
        else:
            pooled = torch.cat([hidden.mean(dim=0), hidden.std(dim=0)])
        vec = pooled.detach().cpu().numpy().astype(float)
        return {f"emb_{i:03d}": float(x) for i, x in enumerate(vec)}

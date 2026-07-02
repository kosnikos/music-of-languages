"""Speech-vs-music tagging via an AudioSet AST classifier (Gong et al. 2021,
arXiv:2104.01778; AudioSet: Gemmeke et al. 2017). Catches sung vocals / music
that the VAD speech gate let through. The scorer is injectable so unit tests
never load the model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
_AST_INPUT_S = 10.0  # AST is trained on ~10 s inputs; window the 30 s and average


@dataclass(frozen=True)
class TaggerScores:
    speech: float
    music: float


def tag_speech_music(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, *, scorer=None
) -> TaggerScores:
    """Speech and music probabilities for `signal`. `music = max(Music, Singing)`."""
    scorer = scorer or _ast_scorer
    probs = scorer(signal, sr)
    speech = float(probs.get("Speech", 0.0))
    music = max(float(probs.get("Music", 0.0)), float(probs.get("Singing", 0.0)))
    return TaggerScores(speech=speech, music=music)


@lru_cache(maxsize=1)
def _load_ast():
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    model = ASTForAudioClassification.from_pretrained(AST_MODEL)
    model.eval()
    fe = AutoFeatureExtractor.from_pretrained(AST_MODEL)
    return model, fe


def _ast_scorer(signal: np.ndarray, sr: int) -> dict[str, float]:
    """Mean per-class sigmoid probability over ~10 s windows of `signal`."""
    import torch

    model, fe = _load_ast()
    win = int(_AST_INPUT_S * sr)
    chunks = [signal[i : i + win] for i in range(0, max(len(signal), 1), win)]
    sums = None
    n = 0
    for ch in chunks:
        if len(ch) < sr:  # skip a <1 s tail
            continue
        inputs = fe(ch, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            probs = torch.sigmoid(model(**inputs).logits)[0]
        sums = probs if sums is None else sums + probs
        n += 1
    if n == 0:
        return {}
    mean = sums / n
    id2label = model.config.id2label
    return {id2label[i]: float(mean[i]) for i in range(mean.shape[0])}

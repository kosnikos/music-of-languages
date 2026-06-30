"""OpenAI Whisper language-ID + transcript (Radford et al. 2022, arXiv:2212.04356).
verbose_json `.language` is a lowercase English name (e.g. 'english'), comparable
directly to the SEED_LANGUAGES keys. Fail-soft: ('', '') on error -> downstream tie-break."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from musiclang.config import TARGET_SAMPLE_RATE

WHISPER_MODEL = "whisper-1"


def transcribe_language(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, *, client=None
) -> tuple[str, str]:
    """Return (detected_language_lowercased, transcript) for `signal`."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "seg.wav"
            sf.write(str(wav), signal, sr)
            with open(wav, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    model=WHISPER_MODEL, file=fh, response_format="verbose_json"
                )
        lang = (getattr(resp, "language", "") or "").lower()
        text = (getattr(resp, "text", "") or "").strip()
        return lang, text
    except Exception:
        return "", ""

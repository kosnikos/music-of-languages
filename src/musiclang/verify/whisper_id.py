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
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, *, client=None, max_retries: int = 3
) -> tuple[str, str]:
    """Return (detected_language_lowercased, transcript) for `signal`."""
    import time
    try:
        from openai import OpenAI
        c = client or OpenAI()
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "seg.wav"
            sf.write(str(wav), signal, sr)
            for attempt in range(max_retries):
                try:
                    with open(wav, "rb") as fh:
                        resp = c.audio.transcriptions.create(
                            model=WHISPER_MODEL, file=fh, response_format="verbose_json"
                        )
                    return (getattr(resp, "language", "") or "").lower(), (getattr(resp, "text", "") or "").strip()
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(1.5 * (attempt + 1))
        return "", ""
    except Exception:
        return "", ""

"""Hybrid segment verifier: tagger -> whisper -> LLM tie-break (short-circuit)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE

_MIN_TRANSCRIPT = 3  # chars; below this a Whisper transcript is treated as garbage -> tie-break


@dataclass(frozen=True)
class Verdict:
    label: str            # target-speech | music | other-language | other
    confidence: float
    detected_language: str
    transcript: str
    tagger_speech: float
    tagger_music: float
    stage_decided: str    # tagger | whisper | llm


def verify_segment(
    signal: np.ndarray,
    sr: int,
    target_language: str,
    *,
    tagger=None,
    whisper=None,
    llm_judge=None,
    music_threshold: float = 0.1,
) -> Verdict:
    """Label `signal` as target-speech / music / other-language / other.

    `tagger(signal, sr) -> TaggerScores`; `whisper(signal, sr) -> (language, transcript)`;
    `llm_judge(transcript, target_language) -> bool`. All injectable; lazy-default to the
    real implementations so unit tests stay offline.
    """
    if tagger is None:
        from musiclang.verify.tagger import tag_speech_music as tagger
    if whisper is None:
        from musiclang.verify.whisper_id import transcribe_language as whisper
    if llm_judge is None:
        from musiclang.verify.llm_judge import judge_transcript as llm_judge

    scores = tagger(signal, sr)
    # Keep only genuinely music-free speech: reject if the AST music score reaches the
    # threshold, regardless of the speech score. A "carpet" (a music bed under a presenter)
    # scores high on BOTH music and speech and is dropped here; clean speech scores music ~0.00.
    if scores.music >= music_threshold:
        return Verdict("music", scores.music, "", "", scores.speech, scores.music, "tagger")

    lang, text = whisper(signal, sr)
    lang = (lang or "").lower()
    text = (text or "").strip()

    if lang == target_language.lower() and len(text) >= _MIN_TRANSCRIPT:
        return Verdict("target-speech", scores.speech, lang, text,
                       scores.speech, scores.music, "whisper")

    fluent = llm_judge(text, target_language)
    if fluent:
        label = "target-speech"
    elif lang and lang != target_language.lower():
        label = "other-language"
    else:
        label = "other"
    return Verdict(label, scores.speech, lang, text, scores.speech, scores.music, "llm")

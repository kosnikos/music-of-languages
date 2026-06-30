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
    music_threshold: float = 0.5,
    speech_threshold: float = 0.3,
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
    if scores.music >= music_threshold and scores.music > scores.speech:
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

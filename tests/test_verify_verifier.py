# tests/test_verify_verifier.py
import numpy as np
from musiclang.verify.verifier import Verdict, verify_segment
from musiclang.verify.tagger import TaggerScores

SIG = np.zeros(16_000, dtype=np.float32)


def _tagger(speech, music):
    return lambda sig, sr: TaggerScores(speech=speech, music=music)


def test_music_rejected_at_tagger_stage():
    v = verify_segment(SIG, 16_000, "english",
                       tagger=_tagger(0.1, 0.9),
                       whisper=lambda s, sr: ("english", "should not run"),
                       llm_judge=lambda t, l: True)
    assert v.label == "music" and v.stage_decided == "tagger"


def test_target_language_accepted_at_whisper_stage():
    v = verify_segment(SIG, 16_000, "german",
                       tagger=_tagger(0.9, 0.1),
                       whisper=lambda s, sr: ("german", "guten tag wie geht es"),
                       llm_judge=lambda t, l: (_ for _ in ()).throw(AssertionError("llm must not run")))
    assert v.label == "target-speech" and v.stage_decided == "whisper"
    assert v.detected_language == "german"


def test_language_mismatch_goes_to_tiebreak_then_other_language():
    v = verify_segment(SIG, 16_000, "english",
                       tagger=_tagger(0.8, 0.1),
                       whisper=lambda s, sr: ("french", "bonjour tout le monde"),
                       llm_judge=lambda t, l: False)
    assert v.label == "other-language" and v.stage_decided == "llm"


def test_tiebreak_rescues_target_speech_on_short_transcript():
    v = verify_segment(SIG, 16_000, "english",
                       tagger=_tagger(0.8, 0.1),
                       whisper=lambda s, sr: ("english", "hi"),  # too short -> tiebreak
                       llm_judge=lambda t, l: True)
    assert v.label == "target-speech" and v.stage_decided == "llm"


def test_empty_transcript_other():
    v = verify_segment(SIG, 16_000, "english",
                       tagger=_tagger(0.6, 0.2),
                       whisper=lambda s, sr: ("", ""),
                       llm_judge=lambda t, l: False)
    assert v.label == "other" and v.stage_decided == "llm"

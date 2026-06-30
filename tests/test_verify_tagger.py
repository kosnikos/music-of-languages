import numpy as np
from musiclang.verify.tagger import TaggerScores, tag_speech_music


def test_tag_extracts_speech_and_max_music_singing():
    fake = lambda sig, sr: {"Speech": 0.91, "Music": 0.10, "Singing": 0.40, "Dog": 0.02}
    s = tag_speech_music(np.zeros(16_000, dtype=np.float32), scorer=fake)
    assert isinstance(s, TaggerScores)
    assert s.speech == 0.91
    assert s.music == 0.40  # max(Music, Singing)


def test_tag_missing_labels_default_zero():
    s = tag_speech_music(np.zeros(10, dtype=np.float32), scorer=lambda sig, sr: {})
    assert s.speech == 0.0 and s.music == 0.0

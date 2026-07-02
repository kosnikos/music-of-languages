import numpy as np
from musiclang.probe.core import (
    RecordingRef, ProbeResult, measure_cleanliness, MIN_CLEAN_SPEECH_S,
)


def test_measure_cleanliness_sums_speech_and_flags_30s():
    signal = np.zeros(16_000 * 40, dtype=np.float32)
    # injected VAD: 20 s + 15 s = 35 s of speech, clears the 30 s bar
    clean_s, meets = measure_cleanliness(
        signal, speech_fn=lambda sig, sr: [(0.0, 20.0), (25.0, 40.0)]
    )
    assert clean_s == 35.0
    assert meets is True


def test_measure_cleanliness_below_threshold_not_flagged():
    signal = np.zeros(10, dtype=np.float32)
    clean_s, meets = measure_cleanliness(signal, speech_fn=lambda sig, sr: [(0.0, 12.0)])
    assert clean_s == 12.0
    assert meets is False
    assert MIN_CLEAN_SPEECH_S == 30.0


def test_recordingref_and_proberesult_fields():
    ref = RecordingRef("radio", "finnish", "yle-1", "hls", "http://x.m3u8")
    assert (ref.source, ref.kind, ref.ref) == ("radio", "hls", "http://x.m3u8")
    res = ProbeResult("radio", "finnish", "yle-1", "hls", True, 33.0, True)
    assert res.capturable and res.meets_30s and res.error == ""

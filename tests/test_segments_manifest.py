from musiclang.ingest.manifest import (
    SEGMENTS_COLUMNS, segments_manifest_dataframe, DROPS_COLUMNS, drops_dataframe,
)


def test_segments_manifest_column_order_and_rows():
    rows = [{
        "segment_id": "english_bbc_ep1", "language": "english", "source": "podcast",
        "channel_id": "bbc-global-news", "recording_ref": "ep1", "recorded_at": "2026-06-30T10:00:00Z",
        "clean_speech_s": 78.6, "path": "data/segments/english/english_bbc_ep1.wav",
        "label": "target-speech", "confidence": 0.93, "detected_language": "english",
        "transcript": "hello", "tagger_speech": 0.9, "tagger_music": 0.1, "stage_decided": "whisper",
    }]
    df = segments_manifest_dataframe(rows)
    assert list(df.columns) == SEGMENTS_COLUMNS
    assert df.loc[0, "label"] == "target-speech" and df.loc[0, "language"] == "english"


def test_drops_dataframe_column_order():
    rows = [{"language": "greek", "source": "radio", "channel_id": "ert-deftero",
             "recording_ref": "cap1", "reason": "music", "detail": "tagger music=0.8"}]
    df = drops_dataframe(rows)
    assert list(df.columns) == DROPS_COLUMNS
    assert df.loc[0, "reason"] == "music"

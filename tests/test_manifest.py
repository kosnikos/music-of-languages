from musiclang.ingest.manifest import MANIFEST_COLUMNS, manifest_dataframe


def test_manifest_dataframe_columns_and_rows():
    rows = [
        {"clip_id": "english_01", "language": "english", "station_name": "BBC",
         "station_url": "http://x", "country": "United Kingdom",
         "recorded_at": "2026-06-25T10:00:00+00:00", "duration_s": 42.0,
         "path": "/data/clips/english/01.wav"},
    ]
    df = manifest_dataframe(rows)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "clip_id"] == "english_01"


def test_manifest_dataframe_empty():
    df = manifest_dataframe([])
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 0

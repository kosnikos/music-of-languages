import pytest
from musiclang import config


def test_eight_seed_languages_present():
    assert len(config.SEED_LANGUAGES) == 8
    assert set(config.SEED_LANGUAGES) == {
        "english", "german", "polish", "french",
        "spanish", "italian", "greek", "finnish",
    }


def test_language_spec_fields():
    english = config.SEED_LANGUAGES["english"]
    assert english.name == "English"
    assert english.iso639_1 == "en"
    assert english.radio_browser_lang == "english"


@pytest.mark.parametrize("key,name,iso639_1,radio_browser_lang", [
    ("english", "English", "en", "english"),
    ("german", "German", "de", "german"),
    ("polish", "Polish", "pl", "polish"),
    ("french", "French", "fr", "french"),
    ("spanish", "Spanish", "es", "spanish"),
    ("italian", "Italian", "it", "italian"),
    ("greek", "Greek", "el", "greek"),
    ("finnish", "Finnish", "fi", "finnish"),
])
def test_all_seed_languages_fields(key, name, iso639_1, radio_browser_lang):
    """Verify all 8 seed languages have correct fields (catches transcription errors)."""
    spec = config.SEED_LANGUAGES[key]
    assert spec.name == name
    assert spec.iso639_1 == iso639_1
    assert spec.radio_browser_lang == radio_browser_lang


def test_target_sample_rate():
    """Assert TARGET_SAMPLE_RATE has correct value and type."""
    assert config.TARGET_SAMPLE_RATE == 16000
    assert isinstance(config.TARGET_SAMPLE_RATE, int)


def test_language_spec_is_accessible():
    """Assert LanguageSpec class is accessible at module scope."""
    assert isinstance(config.LanguageSpec, type)


def test_data_dir_is_path():
    from pathlib import Path
    assert isinstance(config.DATA_DIR, Path)

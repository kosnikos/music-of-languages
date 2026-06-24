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


def test_data_dir_is_path():
    from pathlib import Path
    assert isinstance(config.DATA_DIR, Path)

"""Project-wide configuration: seed languages and paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LanguageSpec:
    name: str               # display name
    iso639_1: str           # 2-letter code
    radio_browser_lang: str # 'language' value used by radio-browser.info


SEED_LANGUAGES: dict[str, LanguageSpec] = {
    "english": LanguageSpec("English", "en", "english"),
    "german":  LanguageSpec("German",  "de", "german"),
    "polish":  LanguageSpec("Polish",  "pl", "polish"),
    "french":  LanguageSpec("French",  "fr", "french"),
    "spanish": LanguageSpec("Spanish", "es", "spanish"),
    "italian": LanguageSpec("Italian", "it", "italian"),
    "greek":   LanguageSpec("Greek",   "el", "greek"),
    "finnish": LanguageSpec("Finnish", "fi", "finnish"),
}

# Repo root is two parents up from this file (src/musiclang/config.py -> repo root).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"

# Audio working format used throughout the pipeline.
TARGET_SAMPLE_RATE: int = 16_000

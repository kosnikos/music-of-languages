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


# ---------------------------------------------------------------------------
# Capital-city data for geographic station selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Capital:
    name: str
    lat: float
    lon: float


CAPITALS: dict[str, Capital] = {
    "english": Capital("London",   51.5074,  -0.1278),
    "german":  Capital("Berlin",   52.5200,  13.4050),
    "polish":  Capital("Warsaw",   52.2297,  21.0122),
    "french":  Capital("Paris",    48.8566,   2.3522),
    "spanish": Capital("Madrid",   40.4168,  -3.7038),
    "italian": Capital("Rome",     41.9028,  12.4964),
    "greek":   Capital("Athens",   37.9838,  23.7275),
    "finnish": Capital("Helsinki", 60.1699,  24.9384),
}

# Search radius around the capital in metres.
CAPITAL_GEO_DISTANCE_M: int = 60_000

# Speech-related tags to union-search near the capital.
SPEECH_TAGS: tuple[str, ...] = (
    "talk",
    "news",
    "information",
    "public radio",
    "spoken word",
    "current affairs",
)

# If fewer than this many unique capital stations are found, fall back to nationwide.
MIN_CAPITAL_STATIONS: int = 3

"""OpenAI-based language guard for radio station filtering.

Provider: OpenAI (gpt-4o-mini via structured outputs).
Purpose : Phase-0 best-effort filter — classifies whether a station
          broadcasts primarily in a target language so foreign-language
          stations (e.g. "BBC Arabic" from an English set) can be dropped.
Design  : Fail-open — any API / key error returns True (keep the station)
          so a transient outage never silently discards the whole sample.
          Country-aware & lenient — station country and listed languages are
          fed to the model, which is instructed to keep the station unless it
          CLEARLY broadcasts in a different language.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"


class LanguageVerdict(BaseModel):
    is_primary_language: bool
    reason: str


def is_in_language(
    station_name: str,
    language: str,
    country: str | None = None,
    station_language: str | None = None,
    client=None,
    cache: dict | None = None,
) -> bool:
    """Return True if *station_name* broadcasts primarily in *language*.

    Parameters
    ----------
    station_name:
        Human-readable station name, e.g. ``"BBC Radio 4"``.
    language:
        Target language, e.g. ``"English"``.
    country:
        Country the station is registered in, e.g. ``"Greece"``.  When
        provided it is passed to the model to reduce false drops for
        stations with generic names.
    station_language:
        Comma-separated language tags from the station record, e.g.
        ``"greek,english"``.  Passed to the model as extra evidence.
    client:
        An ``openai.OpenAI`` instance (or compatible fake). Constructed
        lazily from ``OPENAI_API_KEY`` if *None*.
    cache:
        Optional dict.  Cache hits skip the API call; cache misses are
        stored before returning.  Key format: ``"{name}|{country}|{language}"``.

    Returns
    -------
    bool
        ``True``  → keep the station (primary language match, or fail-open).
        ``False`` → drop the station.
    """
    # --- cache key (country-aware) ---
    key = f"{station_name}|{country or ''}|{language}"

    # --- cache hit ---
    if cache is not None and key in cache:
        return cache[key]

    # --- lazy client construction ---
    if client is None:
        from openai import OpenAI  # imported here so tests never touch it
        client = OpenAI()

    # --- API call with fail-open guard ---
    try:
        response = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You filter radio stations by primary broadcast language "
                        "to remove foreign-language stations. "
                        "Prefer to KEEP a station unless it CLEARLY broadcasts in "
                        "a different language than the target. "
                        "When the station's country is one where the target language "
                        "is widely spoken, or you are unsure, keep it."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target language: {language}. "
                        f"Station: name={station_name!r}, country={country!r}, "
                        f"listed_languages={station_language!r}. "
                        "Set is_primary_language=False ONLY if it clearly broadcasts "
                        "primarily in a DIFFERENT language "
                        "(e.g. 'BBC Arabic' in the United Kingdom broadcasts Arabic, "
                        "not English). Otherwise True."
                    ),
                },
            ],
            response_format=LanguageVerdict,
        )
        verdict = response.choices[0].message.parsed
        if verdict is None:
            # Model issued a refusal — keep the station
            result = True
        else:
            result = verdict.is_primary_language
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "language_filter: API call failed for %r (%s); keeping station (fail-open)",
            station_name,
            exc,
        )
        result = True

    # --- populate cache ---
    if cache is not None:
        cache[key] = result

    return result

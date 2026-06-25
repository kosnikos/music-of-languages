"""OpenAI-based language guard for radio station filtering.

Provider: OpenAI (gpt-4o-mini via structured outputs).
Purpose : Phase-0 best-effort filter — classifies whether a station
          broadcasts primarily in a target language so foreign-language
          stations (e.g. "BBC Arabic" from an English set) can be dropped.
Design  : Fail-open — any API / key error returns True (keep the station)
          so a transient outage never silently discards the whole sample.
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
    client:
        An ``openai.OpenAI`` instance (or compatible fake). Constructed
        lazily from ``OPENAI_API_KEY`` if *None*.
    cache:
        Optional dict keyed by station name.  Cache hits skip the API call;
        cache misses are stored before returning.

    Returns
    -------
    bool
        ``True``  → keep the station (primary language match, or fail-open).
        ``False`` → drop the station.
    """
    # --- cache hit ---
    if cache is not None and station_name in cache:
        return cache[station_name]

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
                        "You classify radio stations by their primary broadcast "
                        "language from the station name."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Does the radio station named '{station_name}' broadcast "
                        f"primarily in {language}? "
                        "Well-known examples: 'BBC Arabic' broadcasts in Arabic "
                        "(not English); 'BBC Afrique' in French. "
                        "Answer about the PRIMARY broadcast language."
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
        cache[station_name] = result

    return result

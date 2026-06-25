"""Tests for the OpenAI-based language filter.

No real network calls are made — tests inject a fake client.
No OPENAI_API_KEY is required to run.
"""

import types
import pytest

from musiclang.ingest.language_filter import LanguageVerdict, is_in_language


# ---------------------------------------------------------------------------
# Helper: build a fake openai client
# ---------------------------------------------------------------------------

def fake_client(verdict=None, raises=None, capture=None):
    """Return a minimal stand-in for an openai.OpenAI() client.

    verdict  – a LanguageVerdict (or None for refusal simulation)
    raises   – an exception class/instance to raise instead of returning
    capture  – if a list, each call to parse() appends the kwargs to it
    """
    class _CC:
        def parse(self, **kw):
            if capture is not None:
                capture.append(kw)
            if raises:
                raise raises
            msg = types.SimpleNamespace(parsed=verdict, refusal=None)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)]
            )

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_CC()))


# ---------------------------------------------------------------------------
# Basic verdict tests (new params are optional — existing behaviour preserved)
# ---------------------------------------------------------------------------

def test_returns_true_when_verdict_is_primary_language():
    """is_in_language returns True when the model says the station broadcasts
    primarily in the target language."""
    verdict = LanguageVerdict(is_primary_language=True, reason="BBC Radio 4 is English")
    client = fake_client(verdict=verdict)
    result = is_in_language("BBC Radio 4", "English", client=client)
    assert result is True


def test_returns_false_when_verdict_is_not_primary_language():
    """is_in_language returns False when the model says the station does NOT
    broadcast primarily in the target language (e.g. BBC Arabic vs English)."""
    verdict = LanguageVerdict(is_primary_language=False, reason="BBC Arabic is Arabic, not English")
    client = fake_client(verdict=verdict)
    result = is_in_language("BBC Arabic", "English", client=client)
    assert result is False


# ---------------------------------------------------------------------------
# Cache tests — NEW key format: "{name}|{country}|{language}"
# ---------------------------------------------------------------------------

def test_cache_hit_returns_cached_value_without_calling_client():
    """When the cache already has the new-format key, parse() is never called."""
    bomb_client = fake_client(raises=RuntimeError("should not be called"))
    cache = {"BBC X|United Kingdom|English": True}
    result = is_in_language(
        "BBC X", "English",
        country="United Kingdom",
        client=bomb_client,
        cache=cache,
    )
    assert result is True


def test_cache_miss_populates_cache_with_new_key():
    """On a cache miss the result is stored under the new key format."""
    verdict = LanguageVerdict(is_primary_language=True, reason="Test station is English")
    client = fake_client(verdict=verdict)
    cache = {}
    result = is_in_language("Test Station FM", "English", country="Greece", client=client, cache=cache)
    assert result is True
    assert "Test Station FM|Greece|English" in cache
    assert cache["Test Station FM|Greece|English"] is True


def test_cache_miss_populates_cache_with_false_and_new_key():
    """Cache is also populated (False) under the new key when result is False."""
    verdict = LanguageVerdict(is_primary_language=False, reason="Not English")
    client = fake_client(verdict=verdict)
    cache = {}
    is_in_language("Radio Exterior", "English", country=None, client=client, cache=cache)
    assert "Radio Exterior||English" in cache
    assert cache["Radio Exterior||English"] is False


def test_old_key_does_not_cause_a_hit_for_new_call():
    """An old-format cache entry (name only) must NOT be treated as a hit
    for a new-format call, so the model is still called."""
    captured = []
    verdict = LanguageVerdict(is_primary_language=True, reason="ok")
    client = fake_client(verdict=verdict, capture=captured)
    cache = {"BBC Arabic": False}          # old-format entry
    result = is_in_language("BBC Arabic", "English", country="United Kingdom", client=client, cache=cache)
    # Model WAS called (no hit on old key)
    assert len(captured) == 1
    # New-format key was stored
    assert "BBC Arabic|United Kingdom|English" in cache


# ---------------------------------------------------------------------------
# Fail-open tests
# ---------------------------------------------------------------------------

def test_fail_open_on_exception():
    """When the client raises any exception, is_in_language returns True
    (keep the station) and does not propagate the exception."""
    bomb_client = fake_client(raises=RuntimeError("network error"))
    result = is_in_language("Mystery Station", "English", client=bomb_client)
    assert result is True


def test_fail_open_on_refusal_parsed_is_none():
    """When parsed is None (model refusal), is_in_language returns True
    (keep the station)."""
    client = fake_client(verdict=None)  # verdict=None simulates refusal
    result = is_in_language("Some Station", "English", client=client)
    assert result is True


# ---------------------------------------------------------------------------
# Country + station_language appear in the request sent to the model
# ---------------------------------------------------------------------------

def test_country_and_language_included_in_messages():
    """The country string and target language must appear in the messages
    sent to the model so the guard can use them."""
    captured = []
    verdict = LanguageVerdict(is_primary_language=True, reason="ok")
    client = fake_client(verdict=verdict, capture=captured)

    is_in_language(
        "Alpha 98.9", "English",
        country="Greece",
        station_language="greek,english",
        client=client,
    )

    assert len(captured) == 1
    # Serialise all message content into one string for easy assertion
    all_text = " ".join(
        m["content"] for m in captured[0]["messages"]
    )
    assert "Greece" in all_text
    assert "English" in all_text


def test_station_language_tag_included_in_messages():
    """The station_language tag string must appear in the messages."""
    captured = []
    verdict = LanguageVerdict(is_primary_language=True, reason="ok")
    client = fake_client(verdict=verdict, capture=captured)

    is_in_language(
        "RID 96.8", "English",
        country="Italy",
        station_language="italian",
        client=client,
    )

    all_text = " ".join(m["content"] for m in captured[0]["messages"])
    assert "italian" in all_text


# ---------------------------------------------------------------------------
# Leniency instruction sanity check
# ---------------------------------------------------------------------------

def test_prompt_instructs_leniency():
    """The system/user messages must contain leniency-related wording so the
    instruction cannot be silently removed without breaking tests."""
    captured = []
    verdict = LanguageVerdict(is_primary_language=True, reason="ok")
    client = fake_client(verdict=verdict, capture=captured)

    is_in_language("Alpha 98.9", "English", country="Greece", client=client)

    all_text = " ".join(m["content"] for m in captured[0]["messages"])
    # At least one leniency keyword must appear in the prompt
    leniency_keywords = {"different", "unsure", "keep", "KEEP"}
    assert any(kw in all_text for kw in leniency_keywords), (
        f"Prompt must contain at least one leniency keyword {leniency_keywords}; got:\n{all_text}"
    )

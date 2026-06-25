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

def fake_client(verdict=None, raises=None):
    """Return a minimal stand-in for an openai.OpenAI() client.

    verdict  – a LanguageVerdict (or None for refusal simulation)
    raises   – an exception class/instance to raise instead of returning
    """
    class _CC:
        def parse(self, **kw):
            if raises:
                raise raises
            msg = types.SimpleNamespace(parsed=verdict, refusal=None)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)]
            )

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_CC()))


# ---------------------------------------------------------------------------
# Basic verdict tests
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
# Cache tests
# ---------------------------------------------------------------------------

def test_cache_hit_returns_cached_value_without_calling_client():
    """When the station name is already in cache, the function returns the
    cached value and never invokes the client (which would raise if called)."""
    # Client raises if parse() is ever called — proves it was not called
    bomb_client = fake_client(raises=RuntimeError("should not be called"))
    cache = {"BBC Arabic": False}
    result = is_in_language("BBC Arabic", "English", client=bomb_client, cache=cache)
    assert result is False


def test_cache_miss_populates_cache():
    """On a cache miss, the result is stored in the cache dict."""
    verdict = LanguageVerdict(is_primary_language=True, reason="Test station is English")
    client = fake_client(verdict=verdict)
    cache = {}
    result = is_in_language("Test Station FM", "English", client=client, cache=cache)
    assert result is True
    assert "Test Station FM" in cache
    assert cache["Test Station FM"] is True


def test_cache_miss_populates_cache_with_false():
    """Cache is also populated when result is False."""
    verdict = LanguageVerdict(is_primary_language=False, reason="Not English")
    client = fake_client(verdict=verdict)
    cache = {}
    is_in_language("Radio Exterior", "English", client=client, cache=cache)
    assert cache["Radio Exterior"] is False


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

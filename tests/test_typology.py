import pytest

from musiclang.validation import typology


def test_reference_covers_seed_languages():
    from musiclang.config import SEED_LANGUAGES
    assert set(typology.RHYTHM_CLASS) == set(SEED_LANGUAGES)


def test_class_separation_positive_when_stress_higher():
    # Stress-timed (english/german) high, syllable-timed (french/spanish) low.
    values = {"english": 60.0, "german": 58.0, "french": 40.0, "spanish": 30.0}
    sep = typology.class_separation(values)
    assert sep == pytest.approx(((60 + 58) / 2) - ((40 + 30) / 2))
    assert sep > 0


def test_spearman_against_reference_perfect_order():
    # Use the reference values themselves -> Spearman 1.0
    values = dict(typology.REFERENCE_NPVI_V)
    r = typology.spearman_against_reference(values)
    assert r == pytest.approx(1.0)

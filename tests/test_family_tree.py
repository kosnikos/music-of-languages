import numpy as np
import pandas as pd

from musiclang.validation.family_tree import mantel_test, reference_distance_matrix


def test_reference_is_symmetric_zero_diag_and_genealogically_sane():
    langs = ["english", "german", "spanish", "italian", "finnish"]
    d = reference_distance_matrix(langs)
    assert list(d.index) == langs
    assert (np.diag(d.to_numpy()) == 0).all()
    np.testing.assert_allclose(d.to_numpy(), d.to_numpy().T)
    # Romance pair closer than Romance-vs-Uralic
    assert d.loc["spanish", "italian"] < d.loc["spanish", "finnish"]
    # Germanic pair closer than Germanic-vs-Uralic
    assert d.loc["english", "german"] < d.loc["english", "finnish"]


def test_mantel_identical_matrices_r_one():
    langs = ["english", "german", "spanish", "italian", "finnish"]
    d = reference_distance_matrix(langs)
    r, p = mantel_test(d, d, permutations=99, seed=0)
    assert r > 0.999
    assert p <= 0.05


def test_mantel_shuffled_is_uncorrelated():
    langs = ["english", "german", "polish", "french", "spanish", "italian"]
    d = reference_distance_matrix(langs)
    shuffled = d.loc[langs[::-1], langs[::-1]]
    shuffled.index = langs
    shuffled.columns = langs
    r, _ = mantel_test(d, shuffled, permutations=99, seed=0)
    assert abs(r) < 0.9

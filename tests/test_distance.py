import numpy as np
import pandas as pd

from musiclang.proximity import distance


def _toy():
    return pd.DataFrame(
        {"a": [0.0, 1.0, 10.0], "b": [0.0, 1.0, 10.0]},
        index=["x", "y", "z"],
    )


def test_standardize_zero_mean_unit_std():
    out = distance.standardize(_toy())
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-9)


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    dm = distance.distance_matrix(distance.standardize(_toy()))
    assert list(dm.index) == ["x", "y", "z"]
    assert np.allclose(np.diag(dm.values), 0.0)
    assert np.allclose(dm.values, dm.values.T)
    # x and y are closer to each other than to z
    assert dm.loc["x", "y"] < dm.loc["x", "z"]


def test_mds_2d_shape():
    dm = distance.distance_matrix(distance.standardize(_toy()))
    coords = distance.mds_2d(dm, seed=0)
    assert list(coords.columns) == ["mds_x", "mds_y"]
    assert list(coords.index) == ["x", "y", "z"]


def test_linkage_matrix_rows():
    dm = distance.distance_matrix(distance.standardize(_toy()))
    Z = distance.linkage_matrix(dm)
    assert Z.shape == (2, 4)  # n-1 merges for 3 leaves

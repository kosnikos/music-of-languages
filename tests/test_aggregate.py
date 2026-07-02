import math

import numpy as np
import pandas as pd

from musiclang.features import aggregate
from musiclang.features.aggregate import aggregate_language_robust


def test_aggregate_language_mean_and_std_ignore_nan():
    vectors = [
        {"a": 1.0, "b": 10.0},
        {"a": 3.0, "b": math.nan},
        {"a": math.nan, "b": 20.0},
    ]
    out = aggregate.aggregate_language(vectors)
    assert out["a_mean"] == 2.0          # mean of [1, 3]
    assert out["b_mean"] == 15.0         # mean of [10, 20]
    assert out["a_std"] == np.std([1.0, 3.0])


def test_build_language_table_shape_and_index():
    per_language = {
        "english": [{"a": 1.0}, {"a": 3.0}],
        "french": [{"a": 5.0}, {"a": 7.0}],
    }
    df = aggregate.build_language_table(per_language)
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["english", "french"]
    assert df.loc["english", "a_mean"] == 2.0
    assert df.loc["french", "a_mean"] == 6.0


def test_aggregate_language_robust_resists_outlier():
    vecs = [{"npvi_v": 50.0}, {"npvi_v": 52.0}, {"npvi_v": 54.0}, {"npvi_v": 1000.0}]
    out = aggregate_language_robust(vecs)
    assert out["npvi_v_median"] == 53.0     # (52+54)/2, unaffected by 1000
    assert out["npvi_v_mad"] == 2.0         # median(|x-53|) = median([3,1,1,947]) = 2
    assert "npvi_v_iqr" in out

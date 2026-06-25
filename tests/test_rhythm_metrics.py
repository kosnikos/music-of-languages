import math

import pytest

from musiclang.features import rhythm_metrics as rm


def test_percent_v():
    # vocalic sum = 3, consonantal sum = 1 -> 75%
    assert rm.percent_v([1.0, 2.0], [1.0]) == pytest.approx(75.0)


def test_percent_v_empty_is_nan():
    assert math.isnan(rm.percent_v([], []))


def test_delta_is_population_sd():
    # durations [1, 3]: mean 2, population variance ((1)+(1))/2 = 1, sd = 1
    assert rm.delta([1.0, 3.0]) == pytest.approx(1.0)


def test_varco_is_cv_times_100():
    # [1, 3]: sd 1, mean 2 -> 50
    assert rm.varco([1.0, 3.0]) == pytest.approx(50.0)


def test_npvi_two_intervals():
    # |d1-d2| / ((d1+d2)/2) * 100, single pair: |1-3|/2 *100 = 100
    assert rm.npvi([1.0, 3.0]) == pytest.approx(100.0)


def test_npvi_equal_intervals_is_zero():
    assert rm.npvi([2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_rpvi_two_intervals():
    # mean(|d_k - d_{k+1}|), single pair |1-3| = 2
    assert rm.rpvi([1.0, 3.0]) == pytest.approx(2.0)


def test_single_interval_pvi_is_nan():
    assert math.isnan(rm.npvi([1.0]))
    assert math.isnan(rm.rpvi([1.0]))


def test_single_interval_delta_varco_is_nan():
    assert math.isnan(rm.delta([1.0]))
    assert math.isnan(rm.varco([1.0]))

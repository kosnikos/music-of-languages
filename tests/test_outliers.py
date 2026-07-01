import numpy as np

from musiclang.validation.outliers import OutlierDetector, OutlierResult, CentroidMADDetector


def test_is_a_detector_and_name():
    det = CentroidMADDetector(threshold=3.5, metric="euclidean")
    assert isinstance(det, OutlierDetector)
    assert det.name == "centroid_mad_euclidean"


def test_centroid_mad_flags_the_far_point():
    X = np.zeros((20, 2), dtype=float)
    X[1:, 0] = np.linspace(-1.0, 1.0, 19)  # 19 inliers spread on a line
    X[0] = [50.0, 50.0]                      # 1 planted far outlier
    res = CentroidMADDetector(threshold=3.5).flag(X)
    assert isinstance(res, OutlierResult)
    assert res.is_outlier[0]
    assert res.scores[0] == res.scores.max()
    assert res.is_outlier.sum() == 1


def test_centroid_mad_cosine_flags_opposite_direction():
    X = np.tile([1.0, 0.1], (10, 1)) + np.linspace(-0.02, 0.02, 10)[:, None]
    X[0] = [-1.0, 0.1]  # opposite direction -> large cosine distance
    res = CentroidMADDetector(threshold=3.5, metric="cosine").flag(X)
    assert res.is_outlier[0]
    assert res.is_outlier.sum() == 1


def test_centroid_mad_all_identical_no_flags():
    res = CentroidMADDetector().flag(np.ones((10, 3)))
    assert res.is_outlier.sum() == 0
    assert res.threshold == 3.5


def test_bad_metric_raises():
    import pytest
    with pytest.raises(ValueError):
        CentroidMADDetector(metric="manhattan")

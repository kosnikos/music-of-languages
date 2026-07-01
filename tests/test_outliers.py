import numpy as np
import pandas as pd

from musiclang.validation.outliers import (
    OutlierDetector,
    OutlierResult,
    CentroidMADDetector,
    IsolationForestDetector,
    detect_language_outliers,
)


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


def test_isolation_forest_flags_extreme_point():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(50, 4))
    X[0] = [20.0, 20.0, 20.0, 20.0]
    res = IsolationForestDetector(contamination=0.05, seed=0).flag(X)
    assert res.is_outlier[0]
    assert res.scores[0] == res.scores.max()
    assert IsolationForestDetector().name == "isolation_forest"


def test_detect_language_outliers_per_language():
    idx = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)]
    data = np.zeros((12, 2))
    data[:6, 0] = np.linspace(-1, 1, 6)
    data[6:, 0] = np.linspace(-1, 1, 6) + 10.0
    data[0] = [50.0, 50.0]     # english outlier
    data[6] = [60.0, 60.0]     # greek outlier
    feat_df = pd.DataFrame(data, index=idx, columns=["f0_mean", "npvi_v"])
    labels = {i: ("english" if i.startswith("a") else "greek") for i in idx}
    out = detect_language_outliers(feat_df, labels, CentroidMADDetector(threshold=3.5), space="prosody")
    assert set(out.columns) == {"segment_id", "language", "detector", "space", "score", "is_outlier"}
    flagged = set(out.loc[out["is_outlier"], "segment_id"])
    assert "a0" in flagged and "b0" in flagged
    assert (out["detector"] == "centroid_mad_euclidean").all()


def test_detect_language_outliers_tiny_language_not_flagged():
    feat_df = pd.DataFrame({"f0_mean": [1.0, 2.0]}, index=["a0", "a1"])
    labels = {"a0": "english", "a1": "english"}  # n=2 < 3
    out = detect_language_outliers(feat_df, labels, CentroidMADDetector(), space="prosody")
    assert out["is_outlier"].sum() == 0

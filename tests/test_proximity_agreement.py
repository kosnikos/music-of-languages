import numpy as np
import pandas as pd
import pytest

from musiclang.validation.proximity_agreement import (
    class_silhouette,
    confound_report,
    within_between_separation,
)


def _separated():
    # two tight classes far apart
    langs = ["a1", "a2", "b1", "b2"]
    coords = {"a1": 0.0, "a2": 0.1, "b1": 10.0, "b2": 10.1}
    d = pd.DataFrame(
        [[abs(coords[i] - coords[j]) for j in langs] for i in langs],
        index=langs, columns=langs,
    )
    labels = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    return d, labels


def test_silhouette_high_when_separated():
    d, labels = _separated()
    assert class_silhouette(d, labels) > 0.8


def test_silhouette_nan_single_class():
    d, labels = _separated()
    one = {k: "A" for k in labels}
    assert np.isnan(class_silhouette(d, one))


def test_within_less_than_between():
    d, labels = _separated()
    sep = within_between_separation(d, labels)
    assert sep["within_mean"] < sep["between_mean"]
    assert sep["gap"] > 0


def test_confound_report_keys():
    d, labels = _separated()
    stations = {"a1": "S1", "a2": "S2", "b1": "S1", "b2": "S2"}
    rep = confound_report(d, labels, stations)
    assert set(rep) == {"language_silhouette", "station_silhouette", "language_gap", "station_gap"}
    # language separates (gap>0); station does not align with the geometry
    assert rep["language_gap"] > rep["station_gap"]

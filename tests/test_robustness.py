import numpy as np
import pandas as pd

from musiclang.validation.robustness import (
    proximity_pipeline, leave_one_station_out, leave_one_segment_out, bootstrap_metric_ci,
)


def _toy():
    # 3 languages x 2 channels x 2 segments; 2 prosody-like features
    rng = np.random.default_rng(0)
    rows_f, rows_p = {}, []
    for li, lang in enumerate(["english", "greek", "polish"]):
        for ci in range(2):
            for si in range(2):
                sid = f"{lang}_{ci}_{si}"
                rows_f[sid] = {"f0_mean": li * 10 + rng.normal(0, 0.1),
                               "npvi_v": li * 5 + rng.normal(0, 0.1)}
                rows_p.append({"segment_id": sid, "language": lang,
                               "channel_id": f"{lang}_ch{ci}", "source": "podcast",
                               "recording_ref": sid})
    feat_df = pd.DataFrame.from_dict(rows_f, orient="index")
    prov_df = pd.DataFrame(rows_p).set_index("segment_id")
    return feat_df, prov_df


def test_proximity_pipeline_prosody_shape():
    feat_df, prov_df = _toy()
    dist = proximity_pipeline(feat_df, prov_df, method="prosody")
    assert sorted(dist.index) == ["english", "greek", "polish"]
    assert dist.shape == (3, 3)
    assert np.allclose(np.diag(dist.values), 0.0)


def test_proximity_pipeline_exclude_changes_geometry():
    feat_df, prov_df = _toy()
    full = proximity_pipeline(feat_df, prov_df, method="prosody")
    dropped = proximity_pipeline(feat_df, prov_df, method="prosody", exclude=["english_0_0"])
    assert full.shape == dropped.shape  # still 3 languages
    assert not np.allclose(full.values, dropped.values)


def test_leave_one_station_out_drops_each_channel():
    feat_df, prov_df = _toy()
    res = leave_one_station_out(feat_df, prov_df, lambda d: float(d.values.sum()), method="prosody")
    assert set(res.columns) == {"channel_id", "n_dropped", "metric"}
    assert len(res) == prov_df["channel_id"].nunique()
    assert (res["n_dropped"] >= 1).all()


def test_leave_one_segment_out_len():
    feat_df, prov_df = _toy()
    res = leave_one_segment_out(feat_df, prov_df, lambda d: float(d.values.sum()), method="prosody")
    assert len(res) == len(feat_df)


def test_bootstrap_ci_deterministic_and_ordered():
    feat_df, prov_df = _toy()
    m = lambda d: float(d.values.sum())
    a = bootstrap_metric_ci(feat_df, prov_df, m, method="prosody", n_boot=50, seed=0)
    b = bootstrap_metric_ci(feat_df, prov_df, m, method="prosody", n_boot=50, seed=0)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]
    assert a["n"] <= 50


def test_proximity_pipeline_ssl_shape():
    rows_f, rows_p = {}, []
    for li, lang in enumerate(["english", "greek", "polish"]):
        for k in range(2):
            sid = f"{lang}_{k}"
            rows_f[sid] = {"emb_000": float(li) + 0.01 * k, "emb_001": 1.0 - 0.01 * k}
            rows_p.append({"segment_id": sid, "language": lang, "channel_id": f"{lang}_ch{k}",
                           "source": "podcast", "recording_ref": sid})
    feat_df = pd.DataFrame.from_dict(rows_f, orient="index")
    prov_df = pd.DataFrame(rows_p).set_index("segment_id")
    dist = proximity_pipeline(feat_df, prov_df, method="ssl")
    assert sorted(dist.index) == ["english", "greek", "polish"]
    assert dist.shape == (3, 3)
    assert np.allclose(np.diag(dist.values), 0.0)
    assert np.allclose(dist.values, dist.values.T)  # symmetric


def test_bootstrap_bad_method_raises():
    import pytest
    feat_df, prov_df = _toy()
    with pytest.raises(ValueError):
        bootstrap_metric_ci(feat_df, prov_df, lambda d: 0.0, method="prosady", n_boot=3)

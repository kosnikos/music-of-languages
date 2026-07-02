import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "run_data_integrity", Path(__file__).resolve().parents[1] / "scripts" / "run_data_integrity.py"
)
rdi = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rdi)
assemble_report = rdi.assemble_report


def _synth():
    langs = ["english", "greek", "polish", "french"]
    rng = np.random.default_rng(0)
    prov, pros, emb = [], {}, {}
    for li, lang in enumerate(langs):
        for k in range(4):
            sid = f"{lang}_{k}"
            prov.append({"segment_id": sid, "language": lang, "channel_id": f"{lang}_c{k % 2}",
                         "source": "podcast", "recording_ref": sid})
            pros[sid] = {"npvi_v": li + rng.normal(0, 0.1), "varco_v": li + rng.normal(0, 0.1)}
            emb[sid] = {"emb_000": float(li) + rng.normal(0, 0.05), "emb_001": rng.normal(0, 0.05)}
    prov_df = pd.DataFrame(prov).set_index("segment_id")
    return pd.DataFrame.from_dict(pros, orient="index"), pd.DataFrame.from_dict(emb, orient="index"), prov_df


def test_assemble_report_has_expected_structure():
    pros, emb16, prov = _synth()
    rep = assemble_report(pros, emb16, prov, phase05={"language_gap": 0.0125, "station_gap": 0.0491},
                           mantel_permutations=50)
    for method in ("prosody", "ssl"):
        assert "confound" in rep[method]
        assert {"full", "excluded", "delta"} <= set(rep[method]["metrics"])
    assert rep["ssl"]["confound"]["phase05"]["station_gap"] == 0.0491
    assert "phase05" not in rep["prosody"]["confound"]


def test_within_between_uses_rhythm_class_not_degenerate():
    """`within_between` must key off RHYTHM_CLASS (shared classes), not per-language identity
    labels — with identity labels every language is its own singleton class, so `within` is
    always empty and `gap`/`ratio` are always NaN (degenerate). english/polish are both
    'stress' in RHYTHM_CLASS, so a real within-class pair exists and `gap` should be finite.
    """
    pros, emb16, prov = _synth()
    rep = assemble_report(pros, emb16, prov, phase05={"language_gap": 0.0125, "station_gap": 0.0491},
                           mantel_permutations=50)
    for method in ("prosody", "ssl"):
        wb = rep[method]["metrics"]["full"]["within_between"]
        assert np.isfinite(wb["within_mean"])
        assert np.isfinite(wb["gap"])


def test_to_native_converts_numpy_and_nan():
    """`_to_native` (the JSON-safety layer) must unwrap numpy scalars to native Python and
    turn NaN into None, recursively through dicts/lists — see its docstring.
    """
    val = rdi._to_native(np.float64(1.5))
    assert val == 1.5
    assert type(val) is float

    assert rdi._to_native(float("nan")) is None

    nested = rdi._to_native({"a": np.float64(2.0), "b": [float("nan"), 3.0]})
    assert nested == {"a": 2.0, "b": [None, 3.0]}

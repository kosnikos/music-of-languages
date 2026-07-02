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
    rep = assemble_report(pros, emb16, prov, phase05={"language_gap": 0.0125, "station_gap": 0.0491})
    for method in ("prosody", "ssl"):
        assert "confound" in rep[method]
        assert {"full", "excluded", "delta"} <= set(rep[method]["metrics"])
    assert rep["prosody"]["confound"]["phase05"]["station_gap"] == 0.0491

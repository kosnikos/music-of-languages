# Data Integrity D+E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the outlier-detection + robust-aggregation strand over the 178 verified independent segments, then re-run the station-vs-language confound check + interim clustering and write the findings doc.

**Architecture:** Extend the existing `musiclang` extractors/proximity/validation with (1) a one-pass multi-layer SSL cache + a direct per-segment feature loader that skips re-cleaning, (2) a swappable `OutlierDetector` ABC mirroring `FeatureExtractor`, (3) robust aggregation + a `proximity_pipeline` that rebuilds per-language geometry from any segment subset for leave-one-station-out / bootstrap stability. A controller-owned background job embeds the segments once; a thin driver assembles the before/after report + figures.

**Tech Stack:** Python 3.11+, numpy/pandas/scipy/scikit-learn, transformers+torch (XLS-R, CPU), matplotlib, pytest (+ monkeypatch), uv.

## Global Constraints

- Package is `musiclang`, src-layout under `src/musiclang/`. Run everything with `uv run` (managed CPython; `python-preference = only-managed`).
- Run tests with `uv run pytest`. Fakes/synthetic by default; real-model paths marked `@pytest.mark.slow` (deselected by `addopts = -m 'not slow'`). No `conftest.py` — fixtures inline.
- Dependencies already present (do NOT add new ones): numpy, pandas, pyarrow, scipy, **scikit-learn>=1.4** (`IsolationForest`), librosa, soundfile, **matplotlib>=3.8**, transformers, torch>=2.2,<2.12.
- Input data is **gitignored** (`data/**`): commit code + docs only, never data/wavs/parquets/figures under `data/`. Figures for the doc go under `docs/figures/data-integrity/` (committed).
- Analysis unit is **one segment per recording** → segments are independent; `channel_id` is the station/confound unit; each channel carries ~3 segments.
- Headline SSL config = XLS-R-300m, **layer 16**, 30 s, mean-pool; cache layers **{12, 16, -1}** in one forward pass. Prosody = the 16 scalars.
- Outlier handling = **report-both** (compute with and without exclusion, show delta). MAD threshold = **3.5**.
- Segments are already clean 30 s — **never re-run `clean_clip`/VAD**; load the wav and extract on the whole signal.
- Conventional-commit messages; end each with the Co-Authored-By trailer used in this repo.

---

### Task 1: One-pass multi-layer SSL extraction

**Files:**
- Modify: `src/musiclang/features/ssl_embedding.py` (add `extract_layers` method to `SSLEmbeddingExtractor`)
- Test: `tests/test_ssl_embedding.py` (add cases; reuse existing `_FakeFeat`/`_FakeModel`/`_patch`)

**Interfaces:**
- Consumes: existing `_load_model` (monkeypatched in tests), `self.pooling`.
- Produces: `SSLEmbeddingExtractor.extract_layers(signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, layers: Sequence[int] = (12, 16, -1)) -> dict[int, FeatureVector]` — one forward pass; `{layer_index: {"emb_000": float, ...}}`. Layer indices match the constructor `layer` arg (index into `hidden_states`; `-1` = last).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ssl_embedding.py` (the module already imports numpy, torch, `ssl_embedding`, `SSLEmbeddingExtractor`, and defines `_patch`):

```python
def test_extract_layers_one_pass_multi_layer(monkeypatch):
    _patch(monkeypatch)  # _FakeModel: 4 layers, hidden=6; hidden_states[k] filled with float k
    ex = SSLEmbeddingExtractor(pooling="mean")
    out = ex.extract_layers(np.zeros(16_000, dtype=np.float32), sr=16_000, layers=(1, 3, -1))
    assert set(out) == {1, 3, -1}
    assert list(out[1]) == [f"emb_{i:03d}" for i in range(6)]
    assert all(v == 1.0 for v in out[1].values())   # layer 1 filled with 1.0
    assert all(v == 3.0 for v in out[3].values())    # layer 3 filled with 3.0
    assert all(v == 3.0 for v in out[-1].values())   # -1 == last == index 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ssl_embedding.py::test_extract_layers_one_pass_multi_layer -v`
Expected: FAIL — `AttributeError: 'SSLEmbeddingExtractor' object has no attribute 'extract_layers'`

- [ ] **Step 3: Write minimal implementation**

Add this method to `SSLEmbeddingExtractor` (below `extract`). It mirrors `extract` but loops the requested layers over a single `model(...)` call:

```python
    def extract_layers(
        self,
        signal: np.ndarray,
        sr: int = TARGET_SAMPLE_RATE,
        layers: "Sequence[int]" = (12, 16, -1),
    ) -> dict[int, FeatureVector]:
        """Pool several hidden layers from ONE forward pass. {layer: FeatureVector}."""
        import torch

        feat, model = _load_model(self.model_id, self.device)
        model_sr = int(getattr(feat, "sampling_rate", TARGET_SAMPLE_RATE))
        if sr != model_sr:
            signal = librosa.resample(signal.astype(np.float32), orig_sr=sr, target_sr=model_sr)
        inputs = feat(signal, sampling_rate=model_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        result: dict[int, FeatureVector] = {}
        for layer in layers:
            hidden = out.hidden_states[layer].squeeze(0)  # (T, H)
            if self.pooling == "mean":
                pooled = hidden.mean(dim=0)
            else:
                pooled = torch.cat([hidden.mean(dim=0), hidden.std(dim=0)])
            vec = pooled.detach().cpu().numpy().astype(float)
            result[layer] = {f"emb_{i:03d}": float(x) for i, x in enumerate(vec)}
        return result
```

Add `from collections.abc import Sequence` to the imports (or keep the string annotation as shown).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ssl_embedding.py -v`
Expected: PASS (all existing + the new test)

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/features/ssl_embedding.py tests/test_ssl_embedding.py
git commit -m "feat(features): one-pass multi-layer SSL extraction (extract_layers)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Direct per-segment feature loader (no re-clean, no re-window)

**Files:**
- Modify: `src/musiclang/pipeline.py` (add `build_segment_features_direct`; `load_audio` is already imported)
- Test: `tests/test_pipeline.py` (add a case; module already imports `pipeline`, `pandas`, `numpy`, defines `_FakeExtractor`)

**Interfaces:**
- Consumes: `load_audio` (module-level import in `pipeline.py`), `FeatureExtractor.extract`, `TARGET_SAMPLE_RATE`.
- Produces: `build_segment_features_direct(manifest: pd.DataFrame, extractor: FeatureExtractor) -> tuple[pd.DataFrame, pd.DataFrame]`. `manifest` requires columns `segment_id, language, channel_id, path` (optional `source`, `recording_ref` carried through). Returns `(prov_df, feat_df)` both indexed by `segment_id`; `prov_df` columns `[language, channel_id, source, recording_ref]`; `feat_df` columns = extractor feature names. One row per segment — no windowing, no `clean_clip`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
from musiclang.pipeline import build_segment_features_direct


def test_build_segment_features_direct_no_reclean(monkeypatch):
    # load_audio is monkeypatched: the loader must NOT call clean_clip / window.
    monkeypatch.setattr(pipeline, "load_audio", lambda path, sr=16_000: np.ones(16_000, dtype=np.float32))
    manifest = pd.DataFrame([{
        "segment_id": "english_seg01", "language": "english", "channel_id": "BBC",
        "source": "podcast", "recording_ref": "ep1", "path": "x.wav",
    }])
    prov_df, feat_df = build_segment_features_direct(manifest, _FakeExtractor())
    assert list(prov_df.index) == ["english_seg01"]
    assert prov_df.loc["english_seg01", "language"] == "english"
    assert prov_df.loc["english_seg01", "channel_id"] == "BBC"
    assert prov_df.loc["english_seg01", "source"] == "podcast"
    assert feat_df.loc["english_seg01", "n"] == 16_000  # whole signal, not a 1s window
    assert len(feat_df) == 1  # exactly one row per segment
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_build_segment_features_direct_no_reclean -v`
Expected: FAIL — `ImportError: cannot import name 'build_segment_features_direct'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/musiclang/pipeline.py`:

```python
def build_segment_features_direct(
    manifest: pd.DataFrame,
    extractor: FeatureExtractor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-segment features for pre-cleaned 30 s wavs: load -> extract on the WHOLE signal.

    Unlike build_segment_features, this does NOT re-clean (VAD) or window — the input
    rows are already independent, verified, clean 30 s segments (one per recording).
    """
    prov_rows: list[dict] = []
    feat_rows: dict[str, dict] = {}
    for _, row in manifest.iterrows():
        signal = load_audio(row["path"])
        if len(signal) == 0:
            continue
        seg_id = row["segment_id"]
        prov_rows.append({
            "segment_id": seg_id,
            "language": row["language"],
            "channel_id": row["channel_id"],
            "source": row.get("source"),
            "recording_ref": row.get("recording_ref"),
        })
        feat_rows[seg_id] = extractor.extract(signal, sr=TARGET_SAMPLE_RATE)
    prov_df = pd.DataFrame(prov_rows).set_index("segment_id")
    feat_df = pd.DataFrame.from_dict(feat_rows, orient="index").sort_index()
    return prov_df, feat_df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): direct per-segment feature loader (no re-clean/window)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Segment embedding cache script + launch as background job

**Files:**
- Create: `scripts/embed_segments.py`
- Test: `tests/test_embed_segments.py` (smoke of the pure aggregation helper on a fake extractor)

**Interfaces:**
- Consumes: `SSLEmbeddingExtractor.extract_layers` (Task 1), the final manifest parquet.
- Produces: `embed_segments(manifest, extractor, layers, out_dir)` writing one parquet per layer:
  `data/segment_embeddings_xlsr_l{12,16,last}.parquet` (index `segment_id`, columns `emb_000…`).
  **Resumable:** skip `segment_id`s already present; append.

- [ ] **Step 1: Write the failing test**

`tests/test_embed_segments.py`:

```python
import numpy as np
import pandas as pd

from scripts.embed_segments import embed_segments, _layer_filename


class _FakeLayerExtractor:
    def extract_layers(self, signal, sr, layers):
        # deterministic per-layer vector; length 2
        return {ly: {"emb_000": float(ly), "emb_001": float(len(signal))} for ly in layers}


def test_layer_filename():
    assert _layer_filename(16).endswith("segment_embeddings_xlsr_l16.parquet")
    assert _layer_filename(-1).endswith("segment_embeddings_xlsr_llast.parquet")


def test_embed_segments_writes_one_parquet_per_layer(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.embed_segments.load_audio",
                        lambda path, sr=16_000: np.ones(100, dtype=np.float32))
    manifest = pd.DataFrame([
        {"segment_id": "en_1", "path": "a.wav"},
        {"segment_id": "el_1", "path": "b.wav"},
    ])
    embed_segments(manifest, _FakeLayerExtractor(), layers=(12, 16, -1), out_dir=tmp_path)
    df16 = pd.read_parquet(tmp_path / "segment_embeddings_xlsr_l16.parquet")
    assert list(df16.index) == ["en_1", "el_1"]
    assert df16.loc["en_1", "emb_000"] == 16.0
    assert df16.loc["en_1", "emb_001"] == 100.0
    # resumable: re-running does not duplicate rows
    embed_segments(manifest, _FakeLayerExtractor(), layers=(12, 16, -1), out_dir=tmp_path)
    assert len(pd.read_parquet(tmp_path / "segment_embeddings_xlsr_l16.parquet")) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embed_segments.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.embed_segments'`

- [ ] **Step 3: Write minimal implementation**

`scripts/embed_segments.py`:

```python
"""Embed the 178 verified segments once with XLS-R (layers {12,16,last}) and cache.

Controller background job: single-process sequential (concurrent CPU torch forward
passes segfaulted in Workstream B), resumable (skips already-cached segment_ids).
Run: uv run python -u scripts/embed_segments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from musiclang.audio import load_audio
from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.features.ssl_embedding import SSLEmbeddingExtractor

DATA_DIR = Path("data/segments")
MANIFEST = DATA_DIR / "segments_manifest_final.parquet"
OUT_DIR = Path("data")
LAYERS = (12, 16, -1)


def _layer_filename(layer: int, out_dir: Path = OUT_DIR) -> str:
    tag = "last" if layer == -1 else str(layer)
    return str(out_dir / f"segment_embeddings_xlsr_l{tag}.parquet")


def _already_done(layers, out_dir) -> set[str]:
    done: set[str] | None = None
    for ly in layers:
        p = Path(_layer_filename(ly, out_dir))
        ids = set(pd.read_parquet(p).index) if p.exists() else set()
        done = ids if done is None else (done & ids)  # done only if in ALL layer files
    return done or set()


def embed_segments(manifest, extractor, layers=LAYERS, out_dir=OUT_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _already_done(layers, out_dir)
    todo = manifest[~manifest["segment_id"].isin(done)]
    per_layer: dict[int, list[dict]] = {ly: [] for ly in layers}
    for n, (_, row) in enumerate(todo.iterrows(), 1):
        signal = load_audio(row["path"], sr=TARGET_SAMPLE_RATE)
        vecs = extractor.extract_layers(signal, sr=TARGET_SAMPLE_RATE, layers=layers)
        for ly in layers:
            per_layer[ly].append({"segment_id": row["segment_id"], **vecs[ly]})
        print(f"[{n}/{len(todo)}] {row['segment_id']}", flush=True)
    for ly in layers:
        if not per_layer[ly]:
            continue
        new = pd.DataFrame(per_layer[ly]).set_index("segment_id")
        p = Path(_layer_filename(ly, out_dir))
        if p.exists():
            new = pd.concat([pd.read_parquet(p), new])
            new = new[~new.index.duplicated(keep="first")]
        new.to_parquet(p)
        print(f"wrote {p} ({len(new)} rows)", flush=True)


def main() -> int:
    manifest = pd.read_parquet(MANIFEST)
    extractor = SSLEmbeddingExtractor(model_id="facebook/wav2vec2-xls-r-300m", pooling="mean")
    embed_segments(manifest, extractor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Ensure `scripts/` is importable: it needs `scripts/__init__.py` (create empty if the repo doesn't have one — check first; the collector `scripts/collect_segments.py` exists so it likely does).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embed_segments.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/embed_segments.py tests/test_embed_segments.py
git commit -m "feat(scripts): resumable one-pass XLS-R segment embedding cache

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: CONTROLLER launches the background job**

After Task 3 review passes, the **controller** (not a task subagent) launches the embedding job in the background and continues to Tasks 5–8 while it runs:

```bash
uv run python -u scripts/embed_segments.py
```

Run it as a background process; monitor its `-u` progress. It must complete (all three `segment_embeddings_xlsr_l*.parquet` present with 178 rows) **before Task 9**. If it dies, re-launch — it resumes from the cache. (~78 s/segment CPU per project memory; single-process by design to avoid the Workstream-B concurrent-torch segfault.)

---

### Task 4: Prosody segment feature table (+ shared provenance)

**Files:**
- Create: `scripts/build_segment_prosody.py`
- Test: `tests/test_build_segment_prosody.py` (smoke on a fake extractor + monkeypatched loader)

**Interfaces:**
- Consumes: `build_segment_features_direct` (Task 2), `ProsodyAcousticExtractor`.
- Produces: `data/segment_features_prosody.parquet` (index `segment_id`, 16 scalar cols) and
  `data/segment_provenance.parquet` (index `segment_id`, cols `language, channel_id, source, recording_ref`) — the provenance table every later task joins against.

- [ ] **Step 1: Write the failing test**

`tests/test_build_segment_prosody.py`:

```python
import numpy as np
import pandas as pd

from scripts import build_segment_prosody as bsp


class _FakeExtractor:
    @property
    def name(self):
        return "fake"

    def extract(self, signal, sr):
        return {"npvi_v": float(signal.mean()), "varco_v": float(len(signal))}


def test_build_writes_prosody_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(bsp.pipeline, "load_audio", lambda path, sr=16_000: np.ones(16_000, dtype=np.float32))
    manifest = pd.DataFrame([
        {"segment_id": "en_1", "language": "english", "channel_id": "BBC",
         "source": "podcast", "recording_ref": "ep1", "path": "a.wav"},
    ])
    bsp.build(manifest, _FakeExtractor(), out_dir=tmp_path)
    feats = pd.read_parquet(tmp_path / "segment_features_prosody.parquet")
    prov = pd.read_parquet(tmp_path / "segment_provenance.parquet")
    assert feats.loc["en_1", "varco_v"] == 16_000
    assert prov.loc["en_1", "channel_id"] == "BBC"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_build_segment_prosody.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module has no attribute 'build'`

- [ ] **Step 3: Write minimal implementation**

`scripts/build_segment_prosody.py`:

```python
"""Prosody (16-scalar) feature table + shared provenance for the 178 segments.

Cheap CPU pass; run synchronously. Run: uv run python scripts/build_segment_prosody.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from musiclang import pipeline
from musiclang.features.prosody_acoustic import ProsodyAcousticExtractor

MANIFEST = Path("data/segments/segments_manifest_final.parquet")
OUT_DIR = Path("data")


def build(manifest, extractor, out_dir=OUT_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_df, feat_df = pipeline.build_segment_features_direct(manifest, extractor)
    feat_df.to_parquet(out_dir / "segment_features_prosody.parquet")
    prov_df.to_parquet(out_dir / "segment_provenance.parquet")
    print(f"prosody: {feat_df.shape}, provenance: {prov_df.shape}", flush=True)


def main() -> int:
    manifest = pd.read_parquet(MANIFEST)
    build(manifest, ProsodyAcousticExtractor())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_build_segment_prosody.py -v`
Expected: PASS

- [ ] **Step 5: Commit + CONTROLLER runs it for real**

```bash
git add scripts/build_segment_prosody.py tests/test_build_segment_prosody.py
git commit -m "feat(scripts): prosody segment table + shared provenance

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Then controller runs `uv run python scripts/build_segment_prosody.py` (seconds/segment) to produce the two parquets used by Tasks 8–9.

---

### Task 5: OutlierDetector ABC + CentroidMADDetector

**Files:**
- Create: `src/musiclang/validation/outliers.py`
- Test: `tests/test_outliers.py`

**Interfaces:**
- Produces:
  - `OutlierResult` (frozen dataclass): `scores: np.ndarray`, `is_outlier: np.ndarray`, `threshold: float`.
  - `OutlierDetector(ABC)`: `name` property; `flag(self, X: np.ndarray) -> OutlierResult` (X = `(n_segments, n_features)` for ONE language).
  - `CentroidMADDetector(threshold: float = 3.5, metric: str = "euclidean")` — `metric` ∈ {`"euclidean"`, `"cosine"`}. Flags points whose distance-to-centroid robust-z (median+MAD) exceeds `threshold` (upper tail only). `MAD == 0` → no flags.

- [ ] **Step 1: Write the failing test**

`tests/test_outliers.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_outliers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'musiclang.validation.outliers'`

- [ ] **Step 3: Write minimal implementation**

`src/musiclang/validation/outliers.py`:

```python
"""Swappable per-language outlier detection (mirrors features/base.FeatureExtractor).

Primary method: robust distance-from-centroid (median + MAD z on a 1-D distance),
which stays stable in 1024-dim embeddings at small n where covariance methods fail.
MAD: Leys et al. 2013, https://doi.org/10.1016/j.jesp.2013.03.013
Isolation Forest: Liu et al. 2008, https://doi.org/10.1109/ICDM.2008.17
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

_MAD_TO_SIGMA = 0.6744897501960817  # Phi^-1(0.75): scales MAD to a std-equivalent


@dataclass(frozen=True)
class OutlierResult:
    scores: np.ndarray       # per-segment robust-z (or anomaly score)
    is_outlier: np.ndarray   # bool mask, same row order as X
    threshold: float


class OutlierDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def flag(self, X: np.ndarray) -> OutlierResult:
        """X = (n_segments, n_features) for ONE language."""


class CentroidMADDetector(OutlierDetector):
    def __init__(self, threshold: float = 3.5, metric: str = "euclidean") -> None:
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")
        self.threshold = threshold
        self.metric = metric

    @property
    def name(self) -> str:
        return f"centroid_mad_{self.metric}"

    def flag(self, X: np.ndarray) -> OutlierResult:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit = X / norms
            centroid = unit.mean(axis=0)
            cnorm = np.linalg.norm(centroid) or 1.0
            centroid = centroid / cnorm
            dist = 1.0 - unit @ centroid
        else:
            centroid = X.mean(axis=0)
            dist = np.linalg.norm(X - centroid, axis=1)
        med = float(np.median(dist))
        mad = float(np.median(np.abs(dist - med)))
        if mad == 0.0:
            return OutlierResult(np.zeros(n), np.zeros(n, dtype=bool), self.threshold)
        z = _MAD_TO_SIGMA * (dist - med) / mad
        return OutlierResult(z, z > self.threshold, self.threshold)  # upper tail: far from centroid
```

Create `src/musiclang/validation/outliers.py` only (the package `validation/__init__.py` already exists).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_outliers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/validation/outliers.py tests/test_outliers.py
git commit -m "feat(validation): OutlierDetector ABC + CentroidMADDetector

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: IsolationForestDetector + per-language orchestrator

**Files:**
- Modify: `src/musiclang/validation/outliers.py`
- Test: `tests/test_outliers.py` (add cases)

**Interfaces:**
- Consumes: `OutlierDetector`, `OutlierResult`, `CentroidMADDetector` (Task 5); `standardize` from `proximity.distance`.
- Produces:
  - `IsolationForestDetector(contamination="auto", seed: int = 0)` — `name == "isolation_forest"`; prosody-space use. `threshold` field = `nan`.
  - `detect_language_outliers(feat_df: pd.DataFrame, labels: dict[str, str], detector: OutlierDetector, space: str) -> pd.DataFrame` with columns `[segment_id, language, detector, space, score, is_outlier]`. `space` ∈ {`"prosody"`, `"ssl"`}: prosody → `standardize` the per-language subset; ssl → use `emb_*` columns raw (the cosine detector normalizes). Languages with <3 segments are emitted as not-outlier.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_outliers.py`:

```python
import pandas as pd
from musiclang.validation.outliers import IsolationForestDetector, detect_language_outliers


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_outliers.py -v`
Expected: FAIL — `ImportError: cannot import name 'IsolationForestDetector'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/musiclang/validation/outliers.py`:

```python
import pandas as pd

from musiclang.proximity.distance import standardize


class IsolationForestDetector(OutlierDetector):
    def __init__(self, contamination="auto", seed: int = 0) -> None:
        self.contamination = contamination
        self.seed = seed

    @property
    def name(self) -> str:
        return "isolation_forest"

    def flag(self, X: np.ndarray) -> OutlierResult:
        from sklearn.ensemble import IsolationForest

        X = np.asarray(X, dtype=float)
        clf = IsolationForest(contamination=self.contamination, random_state=self.seed)
        pred = clf.fit_predict(X)          # -1 outlier, 1 inlier
        scores = -clf.score_samples(X)     # higher = more anomalous
        return OutlierResult(scores, pred == -1, float("nan"))


def detect_language_outliers(
    feat_df: pd.DataFrame,
    labels: dict[str, str],
    detector: OutlierDetector,
    space: str,
) -> pd.DataFrame:
    """Run `detector` per language in the given feature `space` ('prosody'|'ssl')."""
    if space not in ("prosody", "ssl"):
        raise ValueError(f"space must be 'prosody' or 'ssl', got {space!r}")
    rows: list[dict] = []
    for lang in sorted(set(labels.values())):
        ids = [i for i in feat_df.index if labels.get(i) == lang]
        if len(ids) < 3:
            rows += [{"segment_id": s, "language": lang, "detector": detector.name,
                      "space": space, "score": float("nan"), "is_outlier": False} for s in ids]
            continue
        sub = feat_df.loc[ids]
        if space == "prosody":
            X = standardize(sub).to_numpy(dtype=float)
        else:
            X = sub[[c for c in sub.columns if c.startswith("emb_")]].to_numpy(dtype=float)
        res = detector.flag(X)
        for s, score, is_out in zip(ids, res.scores, res.is_outlier):
            rows.append({"segment_id": s, "language": lang, "detector": detector.name,
                         "space": space, "score": float(score), "is_outlier": bool(is_out)})
    return pd.DataFrame(rows)
```

Note the imports (`pandas`, `standardize`) are added here for the appended block; move them to the top of the file if the reviewer prefers (keep the module import-clean).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_outliers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/validation/outliers.py tests/test_outliers.py
git commit -m "feat(validation): IsolationForest detector + per-language orchestrator

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Robust aggregation + per-channel centroid weighting

**Files:**
- Modify: `src/musiclang/features/aggregate.py` (add `aggregate_language_robust`)
- Modify: `src/musiclang/proximity/embedding.py` (add `weighting="channel"` + `channel_col`)
- Test: `tests/test_aggregate.py` (add), `tests/test_embedding_proximity.py` (add — create if absent)

**Interfaces:**
- Produces:
  - `aggregate_language_robust(clip_vectors: list[dict[str, float]]) -> dict[str, float]` — per feature `{k}_median`, `{k}_mad`, `{k}_iqr` (NaN-safe).
  - `language_centroids(..., weighting="channel", channel_col="channel_id")` — per-channel mean first, then per-language mean over channels.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_aggregate.py` (import `aggregate_language_robust`):

```python
from musiclang.features.aggregate import aggregate_language_robust


def test_aggregate_language_robust_resists_outlier():
    vecs = [{"npvi_v": 50.0}, {"npvi_v": 52.0}, {"npvi_v": 54.0}, {"npvi_v": 1000.0}]
    out = aggregate_language_robust(vecs)
    assert out["npvi_v_median"] == 53.0     # (52+54)/2, unaffected by 1000
    assert out["npvi_v_mad"] == 2.0         # median(|x-53|) = median([3,1,1,947]) = 2
    assert "npvi_v_iqr" in out
```

Create/extend `tests/test_embedding_proximity.py`:

```python
import numpy as np
import pandas as pd

from musiclang.proximity.embedding import language_centroids


def test_channel_weighting_balances_uneven_channels():
    rows = [{"language": "english", "channel_id": "X", "clip_id": f"x{i}",
             "emb_000": 1.0, "emb_001": 0.0} for i in range(3)]
    rows.append({"language": "english", "channel_id": "Y", "clip_id": "y0",
                 "emb_000": 0.0, "emb_001": 1.0})
    emb_df = pd.DataFrame(rows)
    cent = language_centroids(emb_df, weighting="channel")
    # channels X and Y weigh equally -> unit([1,0]) & unit([0,1]) averaged = [0.5, 0.5]
    assert np.isclose(cent.loc["english", "emb_000"], 0.5)
    assert np.isclose(cent.loc["english", "emb_001"], 0.5)
    # contrast: flat weighting would give [0.75, 0.25]
    flat = language_centroids(emb_df, weighting="flat")
    assert np.isclose(flat.loc["english", "emb_000"], 0.75)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregate.py tests/test_embedding_proximity.py -v`
Expected: FAIL — `ImportError` / `ValueError: weighting must be 'recording' or 'flat'`

- [ ] **Step 3: Write minimal implementations**

Append to `src/musiclang/features/aggregate.py`:

```python
def aggregate_language_robust(clip_vectors: list[dict[str, float]]) -> dict[str, float]:
    """Per-feature median + MAD + IQR across clips, ignoring NaNs (robust vs aggregate_language)."""
    keys = sorted({k for v in clip_vectors for k in v})
    out: dict[str, float] = {}
    for k in keys:
        values = np.array([v.get(k, np.nan) for v in clip_vectors], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            out[f"{k}_median"] = out[f"{k}_mad"] = out[f"{k}_iqr"] = np.nan
        else:
            med = float(np.median(values))
            out[f"{k}_median"] = med
            out[f"{k}_mad"] = float(np.median(np.abs(values - med)))
            out[f"{k}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    return out
```

In `src/musiclang/proximity/embedding.py`, update the signature + branch:

```python
def language_centroids(
    emb_df: pd.DataFrame,
    group: str = "language",
    recording_col: str = "clip_id",
    weighting: str = "recording",
    channel_col: str = "channel_id",
) -> pd.DataFrame:
    """L2-normalize each segment embedding, then average into per-`group` centroids."""
    if weighting not in ("recording", "flat", "channel"):
        raise ValueError(f"weighting must be 'recording', 'flat', or 'channel', got {weighting!r}")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    x = emb_df[emb_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb_df.copy()
    unit[emb_cols] = x / norms
    if weighting == "flat":
        cent = unit.groupby(group)[emb_cols].mean()
    elif weighting == "recording":
        per_rec = unit.groupby([group, recording_col])[emb_cols].mean()
        cent = per_rec.groupby(level=0).mean()
    else:  # channel
        per_chan = unit.groupby([group, channel_col])[emb_cols].mean()
        cent = per_chan.groupby(level=0).mean()
    return cent.sort_index()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aggregate.py tests/test_embedding_proximity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/features/aggregate.py src/musiclang/proximity/embedding.py tests/test_aggregate.py tests/test_embedding_proximity.py
git commit -m "feat: robust language aggregation (median/MAD/IQR) + per-channel centroids

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Robustness module — proximity_pipeline + stability

**Files:**
- Create: `src/musiclang/validation/robustness.py`
- Test: `tests/test_robustness.py`

**Interfaces:**
- Consumes: `aggregate_language_robust`, `language_centroids` (Task 7); `standardize`, `distance_matrix` from `proximity.distance`.
- Produces:
  - `proximity_pipeline(feat_df, prov_df, method, exclude=None, weighting="channel") -> pd.DataFrame` — per-language×language distance matrix from a segment subset. `method="prosody"` → robust median table → `standardize` → euclidean; `method="ssl"` → `language_centroids(weighting)` → cosine.
  - `leave_one_station_out(feat_df, prov_df, metric_fn, method, weighting="channel") -> pd.DataFrame` (cols `channel_id, n_dropped, metric`).
  - `leave_one_segment_out(feat_df, prov_df, metric_fn, method, weighting="channel") -> pd.DataFrame` (cols `segment_id, metric`).
  - `bootstrap_metric_ci(feat_df, prov_df, metric_fn, method, n_boot=1000, seed=0, weighting="channel", ci=95) -> dict` (`point, lo, hi, n`).

- [ ] **Step 1: Write the failing test**

`tests/test_robustness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_robustness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'musiclang.validation.robustness'`

- [ ] **Step 3: Write minimal implementation**

`src/musiclang/validation/robustness.py`:

```python
"""Robust per-language proximity + stability (leave-one-station-out, bootstrap CIs).

proximity_pipeline rebuilds the per-language geometry from ANY segment subset, so
resampling stability just calls it on held-out / resampled segment sets.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from musiclang.features.aggregate import aggregate_language_robust
from musiclang.proximity.distance import distance_matrix, standardize
from musiclang.proximity.embedding import language_centroids


def proximity_pipeline(
    feat_df: pd.DataFrame,
    prov_df: pd.DataFrame,
    method: str,
    exclude: Iterable[str] | None = None,
    weighting: str = "channel",
) -> pd.DataFrame:
    if method not in ("prosody", "ssl"):
        raise ValueError(f"method must be 'prosody' or 'ssl', got {method!r}")
    keep = [i for i in feat_df.index if exclude is None or i not in set(exclude)]
    feats, prov = feat_df.loc[keep], prov_df.loc[keep]
    if method == "prosody":
        table = {}
        for lang, grp in prov.groupby("language"):
            vecs = [feats.loc[sid].to_dict() for sid in grp.index]
            table[lang] = aggregate_language_robust(vecs)
        tbl = pd.DataFrame.from_dict(table, orient="index").sort_index()
        med = tbl[[c for c in tbl.columns if c.endswith("_median")]]
        return distance_matrix(standardize(med), metric="euclidean")
    emb = feats.join(prov[["language", "channel_id"]])
    emb["clip_id"] = emb.index
    cent = language_centroids(emb, weighting=weighting)
    return distance_matrix(cent, metric="cosine")


def leave_one_station_out(feat_df, prov_df, metric_fn: Callable, method, weighting="channel") -> pd.DataFrame:
    rows = []
    for chan, grp in prov_df.groupby("channel_id"):
        dist = proximity_pipeline(feat_df, prov_df, method, exclude=set(grp.index), weighting=weighting)
        rows.append({"channel_id": chan, "n_dropped": len(grp), "metric": float(metric_fn(dist))})
    return pd.DataFrame(rows)


def leave_one_segment_out(feat_df, prov_df, metric_fn: Callable, method, weighting="channel") -> pd.DataFrame:
    rows = []
    for sid in feat_df.index:
        dist = proximity_pipeline(feat_df, prov_df, method, exclude={sid}, weighting=weighting)
        rows.append({"segment_id": sid, "metric": float(metric_fn(dist))})
    return pd.DataFrame(rows)


def bootstrap_metric_ci(
    feat_df, prov_df, metric_fn: Callable, method,
    n_boot: int = 1000, seed: int = 0, weighting: str = "channel", ci: float = 95,
) -> dict:
    rng = np.random.default_rng(seed)
    by_lang = {lang: list(grp.index) for lang, grp in prov_df.groupby("language")}
    vals: list[float] = []
    for _ in range(n_boot):
        picks: list[str] = []
        for ids in by_lang.values():
            picks += rng.choice(ids, size=len(ids), replace=True).tolist()
        new = [f"b{j}" for j in range(len(picks))]
        f = feat_df.loc[picks].copy(); f.index = new
        p = prov_df.loc[picks].copy(); p.index = new
        try:
            v = float(metric_fn(proximity_pipeline(f, p, method, weighting=weighting)))
        except Exception:
            continue
        if not np.isnan(v):
            vals.append(v)
    arr = np.array(vals, dtype=float)
    lo = float(np.percentile(arr, (100 - ci) / 2))
    hi = float(np.percentile(arr, 100 - (100 - ci) / 2))
    return {"point": float(np.median(arr)), "lo": lo, "hi": hi, "n": int(arr.size)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_robustness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/validation/robustness.py tests/test_robustness.py
git commit -m "feat(validation): proximity_pipeline + leave-one-station-out + bootstrap CIs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Analysis driver — the before/after report + figures

**Files:**
- Create: `scripts/run_data_integrity.py`
- Test: `tests/test_run_data_integrity.py` (smoke of the pure `assemble_report` on synthetic frames)

**Prerequisite:** Task 3's embedding bg job has finished (`data/segment_embeddings_xlsr_l{12,16,last}.parquet`, 178 rows) and Task 4's prosody + provenance parquets exist.

**Interfaces:**
- Consumes: everything above + `validation.proximity_agreement.confound_report`, `class_silhouette`, `within_between_separation`; `validation.family_tree.reference_distance_matrix`, `mantel_test`; `validation.typology.RHYTHM_CLASS`; `proximity.distance.{distance_matrix, standardize, linkage_matrix, mds_2d}`; `proximity.embedding.language_centroids`.
- Produces: `data/data_integrity_results.json`, `data/segment_outliers.parquet`, and figures under `docs/figures/data-integrity/`.

**Design.** Split into a pure `assemble_report(prosody_df, emb16_df, prov_df, phase05)` (returns the results dict; unit-tested on synthetic frames) and thin `load_*` / `make_figures` / `main` I/O wrappers.

The report dict must contain, for **both** methods (`prosody`, `ssl`):
1. **confound** — segment-level `confound_report(dist, language_labels, station_labels)` where segment dist = `distance_matrix(standardize(prosody), "euclidean")` / `distance_matrix(unit_embeddings, "cosine")`, `station_labels` = `channel_id`. Include the Phase-0.5 reference (`{"language_gap": 0.0125, "station_gap": 0.0491}`) for the before/after.
2. **metrics with/without outlier exclusion** — per-language proximity via `proximity_pipeline`; `class_silhouette(dist, RHYTHM_CLASS)`, `within_between_separation`, `mantel_test(dist, reference_distance_matrix(langs))` → report **full** and **outlier-excluded** + delta. Outliers from `detect_language_outliers` (CentroidMAD prosody, CentroidMAD ssl@cosine, IsolationForest prosody); union the flags into `data/segment_outliers.parquet`.
3. **stability** — `leave_one_station_out` spread (min/median/max) + `bootstrap_metric_ci` (n_boot=1000, seed=0) on the headline silhouette + confound gap.
4. **layer_sensitivity** — repeat the ssl confound + silhouette for layers 12 / 16 / last.

Figures: dendrogram (`linkage_matrix`) + language MDS (`mds_2d`, 8 langs) colored by `RHYTHM_CLASS`, both methods; segment MDS colored by language and by `channel_id` (two panels).

- [ ] **Step 1: Write the failing smoke test**

`tests/test_run_data_integrity.py`:

```python
import numpy as np
import pandas as pd

from scripts.run_data_integrity import assemble_report


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_data_integrity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_data_integrity'`

- [ ] **Step 3: Write the implementation**

Implement `scripts/run_data_integrity.py` with `assemble_report(prosody_df, emb16_df, prov_df, phase05)` per the Design above, plus `load_inputs()`, `make_figures(...)`, `write_results(...)`, and `main()`. Key building blocks (all already tested upstream) — the report assembly, per method:

```python
def _segment_confound(seg_dist, prov_df, phase05):
    lang = prov_df["language"].to_dict()
    stat = prov_df["channel_id"].to_dict()
    rep = confound_report(seg_dist, lang, stat)
    rep["phase05"] = phase05
    return rep

def _metrics(dist, langs):
    from musiclang.validation.typology import RHYTHM_CLASS
    r, p = mantel_test(dist, reference_distance_matrix(langs))
    return {"rhythm_silhouette": class_silhouette(dist, RHYTHM_CLASS),
            "within_between": within_between_separation(dist, {l: l for l in dist.index}),
            "mantel_r": r, "mantel_p": p}
```

`assemble_report` builds, for prosody: segment dist = `distance_matrix(standardize(prosody_df), "euclidean")`; for ssl: unit-normalize `emb16_df` then `distance_matrix(..., "cosine")`. It computes outliers via `detect_language_outliers`, then per-language proximity via `proximity_pipeline` on the full set and the outlier-excluded set, and packs `{"confound":..., "metrics": {"full":..., "excluded":..., "delta":...}, "outlier_counts":...}`. `delta` = excluded metric − full metric for each headline scalar. Return `{"prosody": {...}, "ssl": {...}, "layer_sensitivity": {...}}`.

Keep `main()` thin: `load_inputs()` reads the parquets, calls `assemble_report`, `write_results` dumps JSON (convert numpy scalars via `float(...)`), `make_figures` writes PNGs. Guard missing embedding caches with a clear error pointing at Task 3's job.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_data_integrity.py -v`
Expected: PASS

- [ ] **Step 5: Commit, then CONTROLLER runs it for real**

```bash
git add scripts/run_data_integrity.py tests/test_run_data_integrity.py
git commit -m "feat(scripts): data-integrity analysis driver (before/after + figures)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Controller runs `uv run python scripts/run_data_integrity.py` (requires the embedding job done), then eyeballs `data/data_integrity_results.json` + the figures for sanity (finite numbers, 8-language matrices, plausible confound gaps) before Task 10.

---

### Task 10: Findings document

**Files:**
- Create: `docs/data-integrity-findings.md`
- Commit: also `docs/figures/data-integrity/*.png`

**Interfaces:** consumes `data/data_integrity_results.json` + the figures (Task 9). No tests (prose deliverable).

- [ ] **Step 1: Write the doc from the results**

The **controller** writes `docs/data-integrity-findings.md` citing the real numbers from `data_integrity_results.json`. Required sections (per spec §7):
1. **Coverage** — segments per language surviving verification; outlier-flag counts per method/space; es(14)/el(15) carpet-scarce + thin-n caveat.
2. **Before/after metrics** — rhythm-class silhouette, within/between, Mantel r/p for baseline + XLS-R, vs Phase-0.5 §4; **with/without outlier exclusion** (the delta).
3. **Before/after confound** — language vs station gaps vs Phase-0.5's 0.012 / 0.049 — the headline test (did language separation reach/beat station?).
4. **Stability** — leave-one-station-out spread + bootstrap CIs on the headline metrics.
5. **Interim clustering** — embed the dendrogram + MDS figures, both methods; honest captions.
6. **Caveats** — es/el thinness; radio/podcast register; XLS-R still provisional; confound-check strength; layer-sensitivity note.

Follow the tone/structure of `docs/feature-exploration-findings.md`. Every number must trace to the JSON.

- [ ] **Step 2: Commit**

```bash
git add docs/data-integrity-findings.md docs/figures/data-integrity/
git commit -m "docs(di): data-integrity findings — before/after confound + clustering

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (spec §-by-§ → task):
- §4.1 one-pass multi-layer SSL → **T1**. §4.2 direct loader → **T2**. §4.3 embedding cache bg job → **T3**. §4.4 prosody table + provenance → **T4**.
- §5 OutlierDetector ABC + CentroidMAD + IsolationForest + orchestrator → **T5, T6**.
- §6.1 robust aggregation + per-channel weighting → **T7**. §6.2 confound re-test, §6.3 metrics, §6.4 stability, §6.5 report-both, §6.6 clustering → **T8** (pipeline+stability) + **T9** (assembly+figures).
- §7 findings doc → **T10**. All spec sections map to a task.

**Placeholder scan:** T9 intentionally describes `assemble_report`'s body at a higher level than the TDD tasks (it is an integration driver over real data, unit-smoke-tested on synthetic frames); all reusable pieces it calls are fully specified and tested in T1–T8, and its required output keys are pinned by the T9 smoke test. No "TBD/handle errors/similar-to" placeholders elsewhere.

**Type consistency:** `build_segment_features_direct -> (prov_df, feat_df)` indexed by `segment_id` (T2) matches the loads in T3/T4/T8/T9. `OutlierResult(scores, is_outlier, threshold)` used identically in T5/T6/T9. `detect_language_outliers(..., space=...)` columns `[segment_id, language, detector, space, score, is_outlier]` consumed in T9. `proximity_pipeline(feat_df, prov_df, method, exclude, weighting)` signature identical in T8 defs and T9 use. `language_centroids(..., weighting="channel", channel_col)` (T7) matches T8's call. Layer cache filenames `segment_embeddings_xlsr_l{12,16,last}.parquet` consistent T3↔T9. No signature drift found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-data-integrity-de.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration. Fits this plan: the controller also owns the embedding bg job (launched after T3) and the real analysis runs (T4, T9) between subagent tasks.
2. **Inline Execution** — execute tasks in this session with batch checkpoints.

Which approach?

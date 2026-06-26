# Feature Exploration Cycle (SSL vs Prosody) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a configurable SSL-embedding `FeatureExtractor` (headline XLS-R-300m) and the comparison harness to evaluate it head-to-head against the prosody baseline on lightly-expanded radio data.

**Architecture:** New code follows the existing `FeatureExtractor` ABC and adds an embedding-specific proximity path (L2-normalized per-language centroid → cosine) plus method-agnostic validation metrics (rhythm-class silhouette, within/between separation, family-tree Mantel) and a sub-clip→recording→station provenance pipeline so dispersion is not pseudoreplicated. Analysis-window length is configurable and swept over {10s, 30s, full}.

**Tech Stack:** Python 3.11+, numpy/pandas/scipy/scikit-learn, transformers + torch (CPU), librosa, parselmouth, silero-vad, pytest. Run everything via `uv run`.

## Global Constraints

- Run all commands with `uv run` (e.g. `uv run pytest`, `uv run python scripts/...`). Environment is Windows / PowerShell; ffmpeg is on PATH; `OPENAI_API_KEY` is in `.env`.
- Do **not** change the torch pin: `torch>=2.2,<2.12`, `torchaudio>=2.2,<2.12`, served from the PyTorch CPU index (`[tool.uv]` in `pyproject.toml`). `python-preference = "only-managed"` must stay.
- `TARGET_SAMPLE_RATE = 16_000` (from `musiclang.config`) is the working sample rate throughout.
- **Provenance convention (spec §9):** every feature-/metric-computing function MUST cite, in its docstring, a link to the authoritative source for its maths/reasoning (scientific preferred; Wikipedia fallback).
- Notebooks are output-stripped by `nbstripout` — commit them stripped (re-run locally to view figures).
- Tests live under `tests/`; follow the existing style (plain `pytest` functions, `from musiclang...` imports, `pytest-mock`/`monkeypatch` available).
- New per-clip feature/metric functions return plain Python floats / `dict[str, float]` to match the `FeatureVector = dict[str, float]` contract.

---

## File Structure

**Create:**
- `src/musiclang/clean/window.py` — `Window` + `window_signal` (pure windowing).
- `src/musiclang/features/ssl_embedding.py` — `SSLEmbeddingExtractor` (HF SSL, configurable model/layer/pooling).
- `src/musiclang/proximity/embedding.py` — `language_centroids` (L2-normalized centroid aggregation).
- `src/musiclang/validation/proximity_agreement.py` — `class_silhouette`, `within_between_separation`, `confound_report`.
- `src/musiclang/validation/family_tree.py` — `reference_distance_matrix`, `mantel_test`, lineage data.
- `src/musiclang/ingest/manifest.py` — `manifest_dataframe` (clip-provenance table builder).
- `src/musiclang/pipeline.py` — `clean_clip`, `segment_clip`, `build_segment_features` (recorded clips → per-segment feature tables).
- `notebooks/04_ssl_vs_prosody.ipynb` — comparison + length sweep + confound check.
- `docs/feature-exploration-findings.md` — the decision doc (cycle output).
- Test files: `tests/test_window.py`, `tests/test_ssl_embedding.py`, `tests/test_proximity_embedding.py`, `tests/test_proximity_agreement.py`, `tests/test_family_tree.py`, `tests/test_manifest.py`, `tests/test_pipeline.py`.

**Modify:**
- `pyproject.toml` — move `transformers` from the `feasibility` extra into core `dependencies`.
- `scripts/collect_sample.py` — write `data/clips_manifest.parquet` with station provenance; keep existing behavior.

**Reused unchanged:** `proximity/distance.py` (`distance_matrix(metric="cosine")` already supports the embedding path), `features/aggregate.py`, `audio.py`, `clean/vad.py`, `features/prosody_acoustic.py`.

---

### Task 1: Windowing helper

**Files:**
- Create: `src/musiclang/clean/window.py`
- Test: `tests/test_window.py`

**Interfaces:**
- Produces:
  - `class Window(NamedTuple): start_s: float; samples: np.ndarray`
  - `window_signal(signal: np.ndarray, sr: int, length_s: float | None, hop_s: float | None = None, min_s: float | None = None) -> list[Window]` — non-overlapping by default (`hop_s=length_s`); drops a trailing window shorter than `min_s` (default `min_s=length_s`, i.e. only complete windows); `length_s=None` returns one window spanning the whole signal.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_window.py
import numpy as np

from musiclang.clean.window import Window, window_signal


def test_exact_multiple_gives_full_windows():
    sr = 16_000
    sig = np.arange(5 * sr, dtype=np.float32)
    wins = window_signal(sig, sr, length_s=1.0)
    assert len(wins) == 5
    assert [round(w.start_s, 3) for w in wins] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(len(w.samples) == sr for w in wins)
    assert isinstance(wins[0], Window)


def test_short_tail_is_dropped_by_default():
    sr = 16_000
    sig = np.zeros(int(5.4 * sr), dtype=np.float32)  # 0.4s tail
    wins = window_signal(sig, sr, length_s=1.0)
    assert len(wins) == 5  # the 0.4s tail < min_s(=1.0) is dropped


def test_full_length_returns_single_window():
    sr = 16_000
    sig = np.zeros(3 * sr, dtype=np.float32)
    wins = window_signal(sig, sr, length_s=None)
    assert len(wins) == 1
    assert len(wins[0].samples) == 3 * sr


def test_empty_signal_returns_empty():
    assert window_signal(np.zeros(0, dtype=np.float32), 16_000, length_s=1.0) == []
    assert window_signal(np.zeros(0, dtype=np.float32), 16_000, length_s=None) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_window.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'musiclang.clean.window'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/clean/window.py
"""Split a cleaned speech signal into fixed-length analysis windows.

Non-overlapping by default so sub-clips of one recording stay maximally
independent (overlap deepens autocorrelation). Window provenance feeds the
hierarchical sub-clip -> recording -> station aggregation used to avoid
pseudoreplication of within-recording variation.

Reference for fixed-window prosodic analysis of short clips: Deloche et al.
(2024), "Language identification from speech rhythm" — 10s clips suffice for
prosody-based language ID. arXiv:2401.14416
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Window(NamedTuple):
    start_s: float
    samples: np.ndarray


def window_signal(
    signal: np.ndarray,
    sr: int,
    length_s: float | None,
    hop_s: float | None = None,
    min_s: float | None = None,
) -> list[Window]:
    """Cut `signal` into windows of `length_s` seconds.

    `length_s=None` -> a single window spanning the whole signal (the "full"
    sweep value). Otherwise non-overlapping (`hop_s` defaults to `length_s`),
    dropping any trailing window shorter than `min_s` (default `length_s`).
    """
    n = len(signal)
    if n == 0:
        return []
    if length_s is None:
        return [Window(0.0, signal)]
    win = int(round(length_s * sr))
    hop = int(round((hop_s if hop_s is not None else length_s) * sr))
    min_len = win if min_s is None else int(round(min_s * sr))
    out: list[Window] = []
    start = 0
    while start < n:
        seg = signal[start : start + win]
        if len(seg) >= min_len:
            out.append(Window(start / sr, seg))
        start += hop
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_window.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/clean/window.py tests/test_window.py
git commit -m "feat: add fixed-length windowing helper for sub-clip analysis"
```

---

### Task 2: SSL embedding extractor (+ transformers dependency)

**Files:**
- Modify: `pyproject.toml` (move `transformers` into core `dependencies`)
- Create: `src/musiclang/features/ssl_embedding.py`
- Test: `tests/test_ssl_embedding.py`

**Interfaces:**
- Consumes: `FeatureExtractor` ABC (`src/musiclang/features/base.py`), `TARGET_SAMPLE_RATE`.
- Produces:
  - `SSLEmbeddingExtractor(model_id: str = "facebook/wav2vec2-xls-r-300m", layer: int = -1, pooling: str = "mean", device: str = "cpu")`
  - `.name -> str` (e.g. `ssl_wav2vec2-xls-r-300m_l-1_mean`); `.extract(signal, sr) -> dict[str, float]` with keys `emb_000…`. `pooling="mean_std"` doubles the dimension (mean ⊕ std).
  - module fn `_load_model(model_id: str, device: str)` returning `(feature_extractor, model)` — the monkeypatch seam for tests.

- [ ] **Step 1: Move transformers into core dependencies**

In `pyproject.toml`, delete the `feasibility` extra's transformers and add it to `dependencies`:

```toml
# in [project] dependencies, add after "matplotlib>=3.8":
    "transformers>=4.40",  # HF SSL models (wav2vec2 / XLS-R / HuBERT) for SSLEmbeddingExtractor
```

```toml
# in [project.optional-dependencies], change the feasibility line to a comment-only note:
# feasibility probe deps graduated to core (transformers) in the SSL feature-exploration cycle.
```

Then sync: `uv sync`
Expected: resolves and installs `transformers` (already partially present from the feasibility extra).

- [ ] **Step 2: Write the failing test (mocked model — no download)**

```python
# tests/test_ssl_embedding.py
import numpy as np
import torch

from musiclang.features.base import FeatureExtractor
from musiclang.features import ssl_embedding
from musiclang.features.ssl_embedding import SSLEmbeddingExtractor


class _FakeFeat:
    sampling_rate = 16_000

    def __call__(self, signal, sampling_rate, return_tensors):
        return {"input_values": torch.tensor(np.asarray(signal), dtype=torch.float32).reshape(1, -1)}


class _FakeOut:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeModel:
    """hidden_states[k] is a (1, T, H) tensor filled with value k."""

    def __init__(self, n_layers=4, hidden=6, frames=10):
        self.n_layers, self.hidden, self.frames = n_layers, hidden, frames

    def __call__(self, input_values, output_hidden_states):
        hs = tuple(
            torch.full((1, self.frames, self.hidden), float(k)) for k in range(self.n_layers)
        )
        return _FakeOut(hs)


def _patch(monkeypatch):
    monkeypatch.setattr(ssl_embedding, "_load_model", lambda model_id, device: (_FakeFeat(), _FakeModel()))


def test_implements_interface_and_name(monkeypatch):
    _patch(monkeypatch)
    ex = SSLEmbeddingExtractor(model_id="facebook/wav2vec2-xls-r-300m", layer=2, pooling="mean")
    assert isinstance(ex, FeatureExtractor)
    assert ex.name == "ssl_wav2vec2-xls-r-300m_l2_mean"


def test_mean_pooling_selects_layer(monkeypatch):
    _patch(monkeypatch)
    ex = SSLEmbeddingExtractor(layer=2, pooling="mean")
    out = ex.extract(np.zeros(16_000, dtype=np.float32), sr=16_000)
    assert len(out) == 6  # hidden size
    assert list(out) == [f"emb_{i:03d}" for i in range(6)]
    assert all(v == 2.0 for v in out.values())  # layer 2 filled with 2.0


def test_mean_std_pooling_doubles_dim(monkeypatch):
    _patch(monkeypatch)
    ex = SSLEmbeddingExtractor(layer=1, pooling="mean_std")
    out = ex.extract(np.zeros(16_000, dtype=np.float32), sr=16_000)
    assert len(out) == 12  # mean(6) + std(6)
    assert all(v == 1.0 for v in list(out.values())[:6])  # means = layer value
    assert all(v == 0.0 for v in list(out.values())[6:])  # std of constant = 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_ssl_embedding.py -v`
Expected: FAIL with `ImportError` / `AttributeError` (module/class not defined)

- [ ] **Step 4: Write minimal implementation**

```python
# src/musiclang/features/ssl_embedding.py
"""Self-supervised speech-embedding feature extractor (wav2vec2 / XLS-R / HuBERT).

Mean-pools (optionally mean+std) a chosen transformer hidden layer into a fixed
vector per clip. Distance between languages is cosine on the per-language
centroid (see proximity/embedding.py). Pooling/distance choice and models:

- wav2vec 2.0: Baevski et al. (2020), arXiv:2006.11477
- XLS-R (multilingual): Babu et al. (2021), arXiv:2111.09296
- HuBERT: Hsu et al. (2021), arXiv:2106.07447

Mid layers tend to carry the most linguistic/phonetic information, so `layer`
is configurable and swept rather than fixed (see the cycle spec, §3.1).
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.features.base import FeatureExtractor, FeatureVector


@lru_cache(maxsize=2)
def _load_model(model_id: str, device: str):
    """Load and cache (feature_extractor, model). Monkeypatched in unit tests."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    feat = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(False)
    return feat, model


class SSLEmbeddingExtractor(FeatureExtractor):
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xls-r-300m",
        layer: int = -1,
        pooling: str = "mean",
        device: str = "cpu",
    ) -> None:
        if pooling not in ("mean", "mean_std"):
            raise ValueError(f"pooling must be 'mean' or 'mean_std', got {pooling!r}")
        self.model_id = model_id
        self.layer = layer
        self.pooling = pooling
        self.device = device

    @property
    def name(self) -> str:
        short = self.model_id.split("/")[-1]
        return f"ssl_{short}_l{self.layer}_{self.pooling}"

    def extract(self, signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> FeatureVector:
        import torch

        feat, model = _load_model(self.model_id, self.device)
        model_sr = int(getattr(feat, "sampling_rate", TARGET_SAMPLE_RATE))
        if sr != model_sr:
            signal = librosa.resample(signal.astype(np.float32), orig_sr=sr, target_sr=model_sr)
        inputs = feat(signal, sampling_rate=model_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[self.layer].squeeze(0)  # (T, H)
        if self.pooling == "mean":
            pooled = hidden.mean(dim=0)
        else:
            pooled = torch.cat([hidden.mean(dim=0), hidden.std(dim=0)])
        vec = pooled.detach().cpu().numpy().astype(float)
        return {f"emb_{i:03d}": float(x) for i, x in enumerate(vec)}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_ssl_embedding.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Add a slow real-model integration test (optional run)**

```python
# append to tests/test_ssl_embedding.py
import pytest


@pytest.mark.slow
def test_real_xlsr_smoke():
    """Downloads XLS-R-300m (~1.2GB). Run: uv run pytest -m slow"""
    ex = SSLEmbeddingExtractor(model_id="facebook/wav2vec2-base", layer=-1, pooling="mean")
    rng = np.random.default_rng(0)
    out = ex.extract(rng.standard_normal(16_000).astype(np.float32), sr=16_000)
    assert len(out) == 768
    assert all(np.isfinite(v) for v in out.values())
```

Register the marker in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: tests that download models or hit the network (deselected by default)"]
addopts = "-m 'not slow'"
```

- [ ] **Step 7: Run the default suite (slow deselected) and commit**

Run: `uv run pytest tests/test_ssl_embedding.py -v`
Expected: PASS (3 passed, 1 deselected)

```bash
git add pyproject.toml src/musiclang/features/ssl_embedding.py tests/test_ssl_embedding.py
git commit -m "feat: add configurable SSL embedding FeatureExtractor (wav2vec2/XLS-R)"
```

---

### Task 3: Embedding centroid aggregation

**Files:**
- Create: `src/musiclang/proximity/embedding.py`
- Test: `tests/test_proximity_embedding.py`

**Interfaces:**
- Produces: `language_centroids(emb_df: pd.DataFrame, group: str = "language", recording_col: str = "clip_id", weighting: str = "recording") -> pd.DataFrame` — L2-normalizes each segment's `emb_*` columns, then averages. `weighting="recording"` averages per recording first then per language (so a chatty station can't dominate); `weighting="flat"` averages over all segments. Returns a `group`-indexed frame of `emb_*` columns. Feed to existing `distance_matrix(centroids, metric="cosine")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proximity_embedding.py
import numpy as np
import pandas as pd

from musiclang.proximity.embedding import language_centroids


def _df():
    # english: 2 segments from clip A, 1 from clip B; french: 1 segment
    return pd.DataFrame(
        {
            "language": ["english", "english", "english", "french"],
            "clip_id":  ["A", "A", "B", "C"],
            "emb_000":  [1.0, 1.0, 0.0, 0.0],
            "emb_001":  [0.0, 0.0, 1.0, 1.0],
        },
        index=["A_w000", "A_w001", "B_w000", "C_w000"],
    )


def test_unit_normalized_and_indexed_by_language():
    cent = language_centroids(_df())
    assert list(cent.index) == ["english", "french"]
    assert list(cent.columns) == ["emb_000", "emb_001"]
    # french has one unit vector (0,1)
    np.testing.assert_allclose(cent.loc["french"].to_numpy(), [0.0, 1.0])


def test_recording_vs_flat_weighting_differ():
    flat = language_centroids(_df(), weighting="flat")
    rec = language_centroids(_df(), weighting="recording")
    # flat: mean of [(1,0),(1,0),(0,1)] = (0.667, 0.333)
    np.testing.assert_allclose(flat.loc["english"].to_numpy(), [2 / 3, 1 / 3])
    # recording: clipA centroid (1,0), clipB (0,1) -> mean (0.5, 0.5)
    np.testing.assert_allclose(rec.loc["english"].to_numpy(), [0.5, 0.5])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proximity_embedding.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/proximity/embedding.py
"""Per-language centroid aggregation for embedding features.

Embeddings do NOT use the scalar mean+std -> z-score -> Euclidean path; they use
the mean of L2-normalized clip embeddings, compared with cosine distance. This
mirrors how SSL/x-vector language embeddings are pooled and compared (cosine on
mean-pooled hidden states): wav2vec 2.0 (Baevski et al. 2020, arXiv:2006.11477).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def language_centroids(
    emb_df: pd.DataFrame,
    group: str = "language",
    recording_col: str = "clip_id",
    weighting: str = "recording",
) -> pd.DataFrame:
    """L2-normalize each segment embedding, then average into per-`group` centroids."""
    if weighting not in ("recording", "flat"):
        raise ValueError(f"weighting must be 'recording' or 'flat', got {weighting!r}")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    x = emb_df[emb_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb_df.copy()
    unit[emb_cols] = x / norms
    if weighting == "flat":
        cent = unit.groupby(group)[emb_cols].mean()
    else:
        per_rec = unit.groupby([group, recording_col])[emb_cols].mean()
        cent = per_rec.groupby(level=0).mean()
    return cent.sort_index()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proximity_embedding.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/proximity/embedding.py tests/test_proximity_embedding.py
git commit -m "feat: add L2-normalized per-language embedding centroids (cosine path)"
```

---

### Task 4: Proximity-space typology metrics + confound report

**Files:**
- Create: `src/musiclang/validation/proximity_agreement.py`
- Test: `tests/test_proximity_agreement.py`

**Interfaces:**
- Consumes: a square distance `pd.DataFrame` (from `distance_matrix`) and a `labels: dict[str, str]` mapping each index entry to a class.
- Produces:
  - `class_silhouette(dist_df, labels) -> float` (NaN if <2 classes among labeled rows)
  - `within_between_separation(dist_df, labels) -> dict[str, float]` keys: `within_mean`, `between_mean`, `gap`, `ratio`
  - `confound_report(dist_df, language_labels, station_labels) -> dict[str, float]` keys: `language_silhouette`, `station_silhouette`, `language_gap`, `station_gap`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proximity_agreement.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_proximity_agreement.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/validation/proximity_agreement.py
"""Method-agnostic proximity-space agreement metrics.

These operate on a distance matrix + class labels, so they evaluate ANY feature
method (scalar rhythm features or SSL embeddings) on the same footing.

- silhouette: Rousseeuw (1987), J. Comput. Appl. Math. 20:53-65,
  https://doi.org/10.1016/0377-0427(87)90125-7
- within/between class separation is the standard cluster-validity contrast.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def class_silhouette(dist_df: pd.DataFrame, labels: dict[str, str]) -> float:
    """Silhouette of `labels` using the precomputed distance matrix (higher=better)."""
    keys = [k for k in dist_df.index if k in labels]
    y = [labels[k] for k in keys]
    if len(set(y)) < 2 or len(keys) < 3:
        return math.nan
    sub = dist_df.loc[keys, keys].to_numpy(dtype=float)
    try:
        return float(silhouette_score(sub, y, metric="precomputed"))
    except ValueError:
        return math.nan


def within_between_separation(dist_df: pd.DataFrame, labels: dict[str, str]) -> dict[str, float]:
    """Mean within-class vs between-class distance, their gap and ratio."""
    keys = [k for k in dist_df.index if k in labels]
    within: list[float] = []
    between: list[float] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            d = float(dist_df.loc[a, b])
            (within if labels[a] == labels[b] else between).append(d)
    wm = float(np.mean(within)) if within else math.nan
    bm = float(np.mean(between)) if between else math.nan
    gap = bm - wm
    ratio = wm / bm if bm not in (0.0, math.nan) and not math.isnan(bm) else math.nan
    return {"within_mean": wm, "between_mean": bm, "gap": gap, "ratio": ratio}


def confound_report(
    dist_df: pd.DataFrame,
    language_labels: dict[str, str],
    station_labels: dict[str, str],
) -> dict[str, float]:
    """Does the geometry cluster by language or by station/channel?

    If `station_*` separation rivals/exceeds `language_*`, the method may be
    encoding channel rather than the language's sound (cycle spec §3.6).
    """
    return {
        "language_silhouette": class_silhouette(dist_df, language_labels),
        "station_silhouette": class_silhouette(dist_df, station_labels),
        "language_gap": within_between_separation(dist_df, language_labels)["gap"],
        "station_gap": within_between_separation(dist_df, station_labels)["gap"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_proximity_agreement.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/validation/proximity_agreement.py tests/test_proximity_agreement.py
git commit -m "feat: add method-agnostic proximity agreement + confound metrics"
```

---

### Task 5: Family-tree reference + Mantel test

**Files:**
- Create: `src/musiclang/validation/family_tree.py`
- Test: `tests/test_family_tree.py`

**Interfaces:**
- Produces:
  - `reference_distance_matrix(languages: list[str]) -> pd.DataFrame` — symmetric, zero-diagonal genealogical tree-path distances for seed languages.
  - `mantel_test(dist_a: pd.DataFrame, dist_b: pd.DataFrame, method: str = "pearson", permutations: int = 10000, seed: int = 0) -> tuple[float, float]` — `(r, p)`, one-sided permutation p-value on upper-triangle entries (aligns `dist_b` to `dist_a`'s index order).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_family_tree.py
import numpy as np
import pandas as pd

from musiclang.validation.family_tree import mantel_test, reference_distance_matrix


def test_reference_is_symmetric_zero_diag_and_genealogically_sane():
    langs = ["english", "german", "spanish", "italian", "finnish"]
    d = reference_distance_matrix(langs)
    assert list(d.index) == langs
    assert (np.diag(d.to_numpy()) == 0).all()
    np.testing.assert_allclose(d.to_numpy(), d.to_numpy().T)
    # Romance pair closer than Romance-vs-Uralic
    assert d.loc["spanish", "italian"] < d.loc["spanish", "finnish"]
    # Germanic pair closer than Germanic-vs-Uralic
    assert d.loc["english", "german"] < d.loc["english", "finnish"]


def test_mantel_identical_matrices_r_one():
    langs = ["english", "german", "spanish", "italian", "finnish"]
    d = reference_distance_matrix(langs)
    r, p = mantel_test(d, d, permutations=99, seed=0)
    assert r > 0.999
    assert p <= 0.05


def test_mantel_shuffled_is_uncorrelated():
    langs = ["english", "german", "polish", "french", "spanish", "italian"]
    d = reference_distance_matrix(langs)
    shuffled = d.loc[langs[::-1], langs[::-1]]
    shuffled.index = langs
    shuffled.columns = langs
    r, _ = mantel_test(d, shuffled, permutations=99, seed=0)
    assert abs(r) < 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_family_tree.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/validation/family_tree.py
"""Genealogical reference distances + a Mantel test for proximity agreement.

The reference is a transparent tree-path distance over the seed languages'
classification (shared-lineage depth). Classification follows Glottolog
(Hammarström, Forkel, Haspelmath & Bank, https://glottolog.org). A richer
alternative is ASJP lexical distance (Wichmann et al., https://asjp.clld.org) —
noted for a later pass.

Mantel test: Mantel (1967), Cancer Research 27:209-220. The matrix-permutation
p-value compares the observed correlation against correlations under random
relabelings of one matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Lineage from broad family -> language (Glottolog classification).
LINEAGE: dict[str, list[str]] = {
    "english": ["IndoEuropean", "Germanic", "WestGermanic", "English"],
    "german":  ["IndoEuropean", "Germanic", "WestGermanic", "German"],
    "polish":  ["IndoEuropean", "BaltoSlavic", "Slavic", "WestSlavic", "Polish"],
    "french":  ["IndoEuropean", "Italic", "Romance", "WesternRomance", "French"],
    "spanish": ["IndoEuropean", "Italic", "Romance", "WesternRomance", "Spanish"],
    "italian": ["IndoEuropean", "Italic", "Romance", "Italian"],
    "greek":   ["IndoEuropean", "Hellenic", "Greek"],
    "finnish": ["Uralic", "Finnic", "Finnish"],
}


def _tree_distance(a: str, b: str) -> int:
    la, lb = LINEAGE[a], LINEAGE[b]
    shared = 0
    for x, y in zip(la, lb):
        if x == y:
            shared += 1
        else:
            break
    return (len(la) - shared) + (len(lb) - shared)


def reference_distance_matrix(languages: list[str]) -> pd.DataFrame:
    """Square genealogical tree-path distance matrix for `languages`."""
    mat = [[float(_tree_distance(a, b)) for b in languages] for a in languages]
    return pd.DataFrame(mat, index=languages, columns=languages)


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if method == "spearman":
        a, b = rankdata(a), rankdata(b)
    elif method != "pearson":
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    return float(np.corrcoef(a, b)[0, 1])


def mantel_test(
    dist_a: pd.DataFrame,
    dist_b: pd.DataFrame,
    method: str = "pearson",
    permutations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Correlation between two distance matrices + a one-sided permutation p-value."""
    order = list(dist_a.index)
    A = dist_a.loc[order, order].to_numpy(dtype=float)
    B = dist_b.loc[order, order].to_numpy(dtype=float)
    iu = np.triu_indices_from(A, k=1)
    a, b = A[iu], B[iu]
    r_obs = _corr(a, b, method)
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    ge = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        Bp = B[np.ix_(perm, perm)]
        if _corr(a, Bp[iu], method) >= r_obs:
            ge += 1
    p = (ge + 1) / (permutations + 1)
    return r_obs, p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_family_tree.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/validation/family_tree.py tests/test_family_tree.py
git commit -m "feat: add Glottolog reference distances + Mantel test"
```

---

### Task 6: Clip manifest builder + collector provenance

**Files:**
- Create: `src/musiclang/ingest/manifest.py`
- Test: `tests/test_manifest.py`
- Modify: `scripts/collect_sample.py`

**Interfaces:**
- Produces: `manifest_dataframe(rows: list[dict]) -> pd.DataFrame` with columns `clip_id, language, station_name, station_url, country, recorded_at, duration_s, path` (stable order). The collector accumulates one such row per usable clip and writes `data/clips_manifest.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
from musiclang.ingest.manifest import MANIFEST_COLUMNS, manifest_dataframe


def test_manifest_dataframe_columns_and_rows():
    rows = [
        {"clip_id": "english_01", "language": "english", "station_name": "BBC",
         "station_url": "http://x", "country": "United Kingdom",
         "recorded_at": "2026-06-25T10:00:00+00:00", "duration_s": 42.0,
         "path": "/data/clips/english/01.wav"},
    ]
    df = manifest_dataframe(rows)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "clip_id"] == "english_01"


def test_manifest_dataframe_empty():
    df = manifest_dataframe([])
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/ingest/manifest.py
"""Per-clip provenance manifest: language, station, time, path.

Provenance (sub-clip -> recording -> station -> language) lets aggregation
estimate dispersion at the recording/station level rather than over correlated
sub-clips, avoiding pseudoreplication of within-recording variation.
"""

from __future__ import annotations

import pandas as pd

MANIFEST_COLUMNS = [
    "clip_id", "language", "station_name", "station_url",
    "country", "recorded_at", "duration_s", "path",
]


def manifest_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build the clip manifest DataFrame with the canonical column order."""
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire provenance into the collector**

In `scripts/collect_sample.py`: import the builder and `datetime`, accumulate rows, write the manifest. Apply these edits.

Add imports near the top (after the existing `from musiclang.ingest...` imports):

```python
from datetime import datetime, timezone

from musiclang.ingest.manifest import manifest_dataframe
from musiclang.clean.vad import total_speech_seconds  # already imported alongside concat_speech
```

Change `_record_usable` to also collect manifest rows. Replace its signature and body with:

```python
def _record_usable(
    stations: list,
    key: str,
    clips_per_lang: int,
    clip_seconds: int,
    attempted: set,
    extractor: "ProsodyAcousticExtractor",
    manifest_rows: list,
) -> list:
    """Record clips; append manifest rows for usable clips (mutates manifest_rows)."""
    vectors: list[dict[str, float]] = []
    for s in stations:
        if len(vectors) >= clips_per_lang:
            break
        if s.url in attempted:
            continue
        attempted.add(s.url)
        clip_id = f"{key}_{len(attempted):02d}"
        out = DATA_DIR / "clips" / key / f"{len(attempted):02d}.wav"
        try:
            record_clip(s.url, out, duration_s=clip_seconds)
            signal = normalize_loudness(load_audio(out))
            segments = extract_speech(signal)
            speech_s = total_speech_seconds(segments)
            if speech_s < 5.0:
                continue
            speech = concat_speech(signal, segments)
            vectors.append(extractor.extract(speech, sr=TARGET_SAMPLE_RATE))
            manifest_rows.append({
                "clip_id": clip_id, "language": key, "station_name": s.name,
                "station_url": s.url, "country": s.country,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "duration_s": float(speech_s), "path": str(out),
            })
        except Exception as exc:
            print(f"[skip] {key} {s.url}: {exc}")
    return vectors
```

In `collect(...)`, add `manifest_rows: list[dict] = []` next to `per_language = {}`, pass `manifest_rows` into every `_record_usable(...)` call (there are three: capital, nationwide, broad), and after writing `lang_features.parquet` add:

```python
    manifest_dataframe(manifest_rows).to_parquet(DATA_DIR / "clips_manifest.parquet")
    print(f"wrote manifest: {len(manifest_rows)} clips")
```

- [ ] **Step 6: Verify collector still imports and the suite passes**

Run: `uv run python -c "import scripts.collect_sample"` (Expected: no error)
Run: `uv run pytest tests/test_manifest.py -v` (Expected: PASS)

- [ ] **Step 7: Commit**

```bash
git add src/musiclang/ingest/manifest.py tests/test_manifest.py scripts/collect_sample.py
git commit -m "feat: write clip provenance manifest from the collector"
```

---

### Task 7: Segment + feature pipeline

**Files:**
- Create: `src/musiclang/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `window_signal` (Task 1), `clean/vad` + `audio` helpers, any `FeatureExtractor` (prosody baseline OR `SSLEmbeddingExtractor`).
- Produces:
  - `clean_clip(path, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray` (load → normalize → VAD → concat)
  - `segment_clip(clip_id, language, station_name, signal, sr, length_s) -> list[tuple[dict, np.ndarray]]`
  - `build_segment_features(manifest: pd.DataFrame, extractor: FeatureExtractor, length_s: float | None) -> tuple[pd.DataFrame, pd.DataFrame]` — `(segments_df indexed by segment_id with provenance cols, features_df indexed by segment_id with the extractor's feature columns)`. Works for both the scalar baseline and the SSL extractor.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import numpy as np
import pandas as pd

from musiclang import pipeline
from musiclang.pipeline import build_segment_features, segment_clip


class _FakeExtractor:
    @property
    def name(self):
        return "fake"

    def extract(self, signal, sr):
        return {"f0": float(signal.mean()), "n": float(len(signal))}


def test_segment_clip_emits_provenance():
    sig = np.zeros(2 * 16_000, dtype=np.float32)
    segs = segment_clip("english_01", "english", "BBC", sig, 16_000, length_s=1.0)
    assert len(segs) == 2
    meta0, samples0 = segs[0]
    assert meta0["segment_id"] == "english_01_w000"
    assert meta0["clip_id"] == "english_01"
    assert meta0["language"] == "english"
    assert meta0["station_name"] == "BBC"
    assert meta0["window_index"] == 0
    assert len(samples0) == 16_000


def test_build_segment_features(monkeypatch):
    # 2s of constant audio -> 2 one-second windows per clip
    monkeypatch.setattr(pipeline, "clean_clip", lambda path, sr=16_000: np.ones(2 * 16_000, dtype=np.float32))
    manifest = pd.DataFrame(
        [{"clip_id": "english_01", "language": "english", "station_name": "BBC", "path": "x.wav"}]
    )
    seg_df, feat_df = build_segment_features(manifest, _FakeExtractor(), length_s=1.0)
    assert list(seg_df.index) == ["english_01_w000", "english_01_w001"]
    assert seg_df.loc["english_01_w000", "language"] == "english"
    assert list(feat_df.columns) == ["f0", "n"]
    assert feat_df.loc["english_01_w000", "f0"] == 1.0
    assert feat_df.loc["english_01_w000", "n"] == 16_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'musiclang.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/pipeline.py
"""Recorded clips -> cleaned speech -> windowed segments -> per-segment features.

One pipeline serves every FeatureExtractor (scalar prosody or SSL embedding):
the per-segment feature table is extractor-agnostic; only the downstream
aggregation/distance differs (scalar: aggregate+standardize+euclidean; embedding:
centroid+cosine, see proximity/embedding.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from musiclang.audio import load_audio, normalize_loudness
from musiclang.clean.vad import concat_speech, extract_speech
from musiclang.clean.window import window_signal
from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.features.base import FeatureExtractor


def clean_clip(path: str | Path, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load -> loudness-normalize -> VAD -> concatenate speech into one array."""
    signal = normalize_loudness(load_audio(path, sr=sr))
    segments = extract_speech(signal, sr=sr)
    return concat_speech(signal, segments, sr=sr)


def segment_clip(
    clip_id: str,
    language: str,
    station_name: str,
    signal: np.ndarray,
    sr: int,
    length_s: float | None,
) -> list[tuple[dict, np.ndarray]]:
    """Window a cleaned clip into (provenance-meta, samples) pairs."""
    out: list[tuple[dict, np.ndarray]] = []
    for i, w in enumerate(window_signal(signal, sr, length_s)):
        meta = {
            "segment_id": f"{clip_id}_w{i:03d}",
            "clip_id": clip_id,
            "language": language,
            "station_name": station_name,
            "window_index": i,
            "start_s": w.start_s,
            "length_s": len(w.samples) / sr,
        }
        out.append((meta, w.samples))
    return out


def build_segment_features(
    manifest: pd.DataFrame,
    extractor: FeatureExtractor,
    length_s: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (segments provenance, per-segment features) for every clip in `manifest`."""
    seg_rows: list[dict] = []
    feat_rows: dict[str, dict] = {}
    for _, row in manifest.iterrows():
        signal = clean_clip(row["path"])
        if len(signal) == 0:
            continue
        for meta, samples in segment_clip(
            row["clip_id"], row["language"], row["station_name"],
            signal, TARGET_SAMPLE_RATE, length_s,
        ):
            seg_rows.append(meta)
            feat_rows[meta["segment_id"]] = extractor.extract(samples, sr=TARGET_SAMPLE_RATE)
    seg_df = pd.DataFrame(seg_rows).set_index("segment_id")
    feat_df = pd.DataFrame.from_dict(feat_rows, orient="index").sort_index()
    return seg_df, feat_df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite (regression gate) and commit**

Run: `uv run pytest`
Expected: PASS (all tests; slow deselected)

```bash
git add src/musiclang/pipeline.py tests/test_pipeline.py
git commit -m "feat: add clip->segment->feature pipeline (extractor-agnostic)"
```

---

### Task 8: Light corpus expansion (data run)

**Files:**
- No code changes (uses Task 6's collector). Produces data artifacts.

This task is an **operational run**, not a unit test. It depends on network access, ffmpeg, and `OPENAI_API_KEY` in `.env`. Partial coverage is acceptable and must be recorded (some stations geo-block/HLS-fail — that is itself a finding, per `docs/phase0-findings.md`).

- [ ] **Step 1: Back up the existing tiny corpus pointer**

Run: `uv run python -c "import pandas as pd; print(pd.read_parquet('data/lang_features.parquet').shape)"`
(Records the prior state for comparison; if the file is absent that's fine.)

- [ ] **Step 2: Run the expanded collection**

Run: `uv run python scripts/collect_sample.py --clips-per-lang 15 --clip-seconds 60`
Expected: per-language `N usable clips` lines, then `wrote manifest: <N> clips`, and on disk:
- `data/clips/<lang>/*.wav` (more clips than before)
- `data/clips_manifest.parquet`
- `data/lang_features.parquet`

- [ ] **Step 3: Verify provenance + coverage**

Run:
```bash
uv run python -c "import pandas as pd; m=pd.read_parquet('data/clips_manifest.parquet'); print(m.groupby('language').agg(clips=('clip_id','size'), stations=('station_name','nunique')))"
```
Expected: a per-language table of clip and **distinct-station** counts. Note any language still thin (<5 clips or 1 station) — this is recorded in the findings doc (Task 10). Do **not** fail the task on thin languages; record them.

- [ ] **Step 4: Commit the manifest (data WAVs stay gitignored)**

```bash
git add -f data/clips_manifest.parquet
git commit -m "data: expanded radio corpus manifest (light expansion)"
```
(If `data/` is gitignored, `-f` force-adds just the small manifest. Confirm WAVs are NOT staged: `git status --short`.)

---

### Task 9: Comparison notebook (SSL vs baseline, length sweep, confound)

**Files:**
- Create: `notebooks/04_ssl_vs_prosody.ipynb`

This notebook is the analysis deliverable; its "test" is running top-to-bottom without error and producing the comparison table/figures. Build it cell-by-cell, executing as you go.

- [ ] **Step 1: Setup + load manifest cell**

```python
import numpy as np, pandas as pd
from pathlib import Path
from musiclang.config import DATA_DIR
from musiclang.features.prosody_acoustic import ProsodyAcousticExtractor
from musiclang.features.ssl_embedding import SSLEmbeddingExtractor
from musiclang.pipeline import build_segment_features
from musiclang.features.aggregate import build_language_table
from musiclang.proximity.distance import standardize, distance_matrix, linkage_matrix, mds_2d
from musiclang.proximity.embedding import language_centroids
from musiclang.validation.proximity_agreement import class_silhouette, within_between_separation, confound_report
from musiclang.validation.family_tree import reference_distance_matrix, mantel_test
from musiclang.validation.typology import RHYTHM_CLASS

manifest = pd.read_parquet(DATA_DIR / "clips_manifest.parquet")
languages = sorted(manifest["language"].unique())
manifest.groupby("language").agg(clips=("clip_id","size"), stations=("station_name","nunique"))
```

- [ ] **Step 2: Helper to score one method's per-language distance matrix**

```python
def score_distance(dist_df):
    labels = {l: RHYTHM_CLASS[l] for l in dist_df.index if l in RHYTHM_CLASS}
    ref = reference_distance_matrix(list(dist_df.index))
    r, p = mantel_test(dist_df, ref, permutations=10_000, seed=0)
    sep = within_between_separation(dist_df, labels)
    return {"silhouette": class_silhouette(dist_df, labels),
            "within_between_gap": sep["gap"], "mantel_r": r, "mantel_p": p}

def baseline_distance(length_s):
    seg, feat = build_segment_features(manifest, ProsodyAcousticExtractor(), length_s)
    feat = feat.join(seg["language"])
    per_lang = {l: [r.drop(labels="language").to_dict() for _, r in g.iterrows()]
                for l, g in feat.groupby("language")}
    table = build_language_table(per_lang)
    return distance_matrix(standardize(table)), seg, feat

def ssl_distance(length_s, layer, model_id="facebook/wav2vec2-xls-r-300m"):
    ex = SSLEmbeddingExtractor(model_id=model_id, layer=layer, pooling="mean")
    seg, emb = build_segment_features(manifest, ex, length_s)
    emb = emb.join(seg[["language", "clip_id"]])
    cache = DATA_DIR / f"clip_embeddings_{ex.name}_{length_s}.parquet"
    emb.to_parquet(cache)
    cent = language_centroids(emb)
    return distance_matrix(cent, metric="cosine"), seg, emb
```

- [ ] **Step 3: Length sweep × method × layer sweep cell**

```python
rows = []
for length_s in [10.0, 30.0, None]:  # None == full clip
    bd, _, _ = baseline_distance(length_s)
    rows.append({"method": "prosody_baseline", "length_s": length_s, "layer": None, **score_distance(bd)})
    for layer in [8, 12, 16, 20, -1]:
        sd, _, _ = ssl_distance(length_s, layer)
        rows.append({"method": "xlsr", "length_s": length_s, "layer": layer, **score_distance(sd)})
results = pd.DataFrame(rows)
results.sort_values(["method", "length_s", "silhouette"], ascending=[True, True, False])
```

- [ ] **Step 4: Confound check cell (best SSL config)**

```python
best = results[results.method == "xlsr"].sort_values("silhouette", ascending=False).iloc[0]
sd, seg, emb = ssl_distance(best.length_s, int(best.layer))
# segment-level cosine distances + segment->language / segment->station labels
seg_dist = distance_matrix(
    emb.set_index(emb.index)[[c for c in emb.columns if c.startswith("emb_")]]
       .pipe(lambda d: d.div(np.linalg.norm(d.values, axis=1, keepdims=True).clip(min=1e-9), axis=0)),
    metric="cosine",
)
lang_labels = seg["language"].to_dict()
station_labels = seg["station_name"].to_dict()
confound_report(seg_dist, lang_labels, station_labels)
```

- [ ] **Step 5: Figures cell (dendrogram + MDS colored by class, best of each method)**

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_method(dist_df, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    dendrogram(linkage_matrix(dist_df), labels=list(dist_df.index), ax=ax1)
    ax1.set_title(f"{title} — dendrogram")
    coords = mds_2d(dist_df)
    for l, (x, y) in coords.iterrows():
        ax2.scatter(x, y); ax2.annotate(l, (x, y))
    ax2.set_title(f"{title} — MDS"); plt.tight_layout(); plt.show()

plot_method(baseline_distance(None)[0], "prosody baseline (full)")
plot_method(ssl_distance(best.length_s, int(best.layer))[0], f"XLS-R l{int(best.layer)}")
```

- [ ] **Step 6: Within-recording stability cell (length sweep robustness)**

```python
def within_recording_stability(length_s):
    _, feat = build_segment_features(manifest, ProsodyAcousticExtractor(), length_s), None
    seg, feat = build_segment_features(manifest, ProsodyAcousticExtractor(), length_s)
    feat = feat.join(seg["clip_id"])
    # std of npvi_v across windows of the same recording, averaged
    return feat.groupby("clip_id")["npvi_v"].std().mean()

pd.DataFrame({"length_s": [10.0, 30.0], "mean_within_recording_npvi_std":
              [within_recording_stability(10.0), within_recording_stability(30.0)]})
```

- [ ] **Step 7: Run the whole notebook, then commit (stripped)**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_ssl_vs_prosody.ipynb`
Expected: executes without error (first run downloads XLS-R-300m).
Then: `uv run nbstripout notebooks/04_ssl_vs_prosody.ipynb`

```bash
git add notebooks/04_ssl_vs_prosody.ipynb
git commit -m "feat: SSL-vs-prosody comparison notebook (length sweep + confound)"
```

---

### Task 10: Decision doc

**Files:**
- Create: `docs/feature-exploration-findings.md`

- [ ] **Step 1: Write the findings doc from the actual notebook numbers**

Capture, from Task 9's `results` table and confound report:
- The corpus coverage after expansion (per-language clips + distinct stations, from Task 8 Step 3).
- The head-to-head table: silhouette, within/between gap, Mantel (r, p) for baseline vs XLS-R across {10s, 30s, full} and the layer sweep — **with the actual numbers**, not placeholders.
- The best XLS-R layer/length and whether it beats the baseline on (a) class silhouette, (b) Mantel r.
- The confound result: language vs station silhouette/gap, and the verdict (does language clustering dominate channel?).
- Within-recording stability vs window length.
- **The decision:** which method carries into Phase 1 (baseline, XLS-R, or "needs the deferred robustness work first"), and the recommended window length for Phase-1 ingest.
- A short "deferred to the robustness cycle" section (outlier filtering, robust aggregation, leave-one-clip-out, Validator ABC, ECAPA).

Follow the structure/voice of `docs/phase0-findings.md` (numbered sections, every figure traceable to an artifact).

- [ ] **Step 2: Commit**

```bash
git add docs/feature-exploration-findings.md
git commit -m "docs: feature-exploration findings — SSL vs prosody verdict"
```

---

## Self-Review

**Spec coverage:**
- §3.1 SSL extractor (configurable model/layer/pooling, citations) → Task 2. ✓
- §3.2 windowing + provenance (manifest + segments) → Tasks 1, 6, 7. ✓
- §3.3 embedding centroid + cosine + caching → Task 3 (centroid), Task 9 Step 2 (cosine via `distance_matrix`, cache parquet). ✓
- §3.4 proximity_agreement + family_tree/Mantel + Glottolog reference → Tasks 4, 5. ✓
- §3.5 light corpus expansion + station provenance → Tasks 6, 8. ✓
- §3.6 confound check → Task 4 (`confound_report`), Task 9 Step 4. ✓
- §3.7 length mini-sweep + within-recording stability → Task 9 Steps 3, 6. ✓
- §4 evaluation protocol / success criteria → Task 9 (scoring) + Task 10 (verdict). ✓
- §6 testing strategy (mocked SSL, pure window/metric tests, slow integration) → Tasks 1–7. ✓
- §7 transformers → core dep → Task 2 Step 1. ✓
- §9 deliverables (all modules, notebook, decision doc) → Tasks 1–10. ✓

**Placeholder scan:** No "TBD/TODO/handle edge cases" left; every code step shows complete code. Task 10 explicitly forbids placeholder numbers (write actual results).

**Type consistency:** `build_segment_features -> (seg_df, feat_df)` consumed in Task 9 with `seg["language"]`/`seg["station_name"]` (cols emitted by `segment_clip` in Task 7). `language_centroids` consumes `emb_*` + `language` + `clip_id` (provided by joining `seg` in Task 9 Step 2). `distance_matrix(metric="cosine")` exists in `proximity/distance.py` (verified). `class_silhouette`/`within_between_separation`/`confound_report` signatures match their Task 9 calls. `mantel_test -> (r, p)` matches `score_distance`. Window `length_s=None` ("full") flows consistently through `window_signal` (Task 1) → `segment_clip`/`build_segment_features` (Task 7) → notebook sweep (Task 9).

---

## Execution

Recommended: **subagent-driven development** — fresh subagent per task with a two-stage review between tasks. Tasks 1–7 are pure TDD and independently gateable; Task 8 is a data run (network-dependent); Tasks 9–10 are analysis/write-up that depend on the expanded corpus.

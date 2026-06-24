# Phase 0 — Exploration & Method Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a minimal, modular Python codebase that fetches real-radio speech for 8 seed languages, extracts alignment-free prosody/rhythm features, and evaluates whether those features reproduce known rhythm typology — producing a written decision on which feature method(s) to carry into Phase 1.

**Architecture:** A small `mol` package with focused modules behind a pluggable `FeatureExtractor` interface. Pure-math units (rhythm-metric formulas, aggregation, distances, validation) are fully unit-tested; audio-dependent units (F0, syllable nuclei, VAD) get deterministic-logic tests plus light synthetic-signal checks. Exploration and the gate decision live in notebooks + a findings doc, not in tested library code. Data ingestion/cleaning are deliberately *lightweight, portable helpers* (not the production pipeline — see `docs/phase1-handoff.md`).

**Tech Stack:** Python 3.11+, numpy/pandas/scipy/scikit-learn, praat-parselmouth, librosa + soundfile, silero-vad (torch), requests (radio-browser.info API), matplotlib, pytest + pytest-mock.

## Global Constraints

- **Python:** 3.11+.
- **Package layout:** `src/`-layout, importable as `mol` (Music Of Languages). Installed editable: `pip install -e .`.
- **Primary feature axis:** prosody & melody, **alignment-free** (no forced alignment / no MFA in Phase 0).
- **Modularity:** every feature method implements the `FeatureExtractor` ABC (`src/mol/features/base.py`).
- **Proximity is continuous:** never assume discrete rhythm classes in code; classes are only a *validation reference*.
- **Data helpers are prototypes:** keep `ingest`/`clean` minimal and behind stable signatures so Phase 1 hardens them in place. NOT in Phase 0: diarization, ad/jingle removal, scheduling/retries, scaling, the retention abstraction.
- **Music avoidance in Phase 0:** prefer **talk/news** stations (tag-filtered) to minimize music; rely on VAD for speech extraction. Robust music removal is Phase 1.
- **Seed languages (exact):** English, German, Polish, French, Spanish, Italian, Greek, Finnish.
- **Storage:** plain local disk under `data/` (gitignored). No retention policy in Phase 0.
- **Testing:** TDD. Pure functions get exact-value tests; audio functions get logic + synthetic-signal tests. Network/subprocess/model calls are mocked in unit tests.
- **Commits:** frequent, one per task minimum. Do not sign commits unless asked.

## File Structure

```
pyproject.toml                          # package metadata + pinned deps
README.md                               # how to set up & run
src/mol/__init__.py
src/mol/config.py                       # seed languages, paths, constants
src/mol/audio.py                        # load_audio, normalize_loudness
src/mol/ingest/__init__.py
src/mol/ingest/radio.py                 # find_stations, record_clip (radio-browser + ffmpeg)
src/mol/clean/__init__.py
src/mol/clean/vad.py                    # extract_speech (silero-vad) + merge logic
src/mol/features/__init__.py
src/mol/features/base.py                # FeatureExtractor ABC + FeatureVector type
src/mol/features/rhythm_metrics.py      # PURE math: pct_v, delta, varco, npvi, rpvi
src/mol/features/intervals.py           # automatic vocalic/consonantal interval detection
src/mol/features/pitch.py               # F0 / intonation features (parselmouth)
src/mol/features/speech_rate.py         # syllable-nuclei speech-rate (De Jong-Wempe style)
src/mol/features/prosody_acoustic.py    # the alignment-free FeatureExtractor (assembles all)
src/mol/features/aggregate.py           # per-language aggregation (mean + dispersion)
src/mol/proximity/__init__.py
src/mol/proximity/distance.py           # standardize, distance matrix, clustering, MDS
src/mol/validation/__init__.py
src/mol/validation/typology.py          # reference class labels/values + agreement metrics
notebooks/01_explore_features.ipynb     # fetch sample, extract, plot rhythm space (deliverable)
notebooks/02_validate_typology.ipynb    # typology agreement + dendrogram (deliverable)
notebooks/03_heavy_methods_feasibility.ipynb  # MFA + embedding feasibility (deliverable)
tests/test_*.py                         # one test module per source module
docs/phase0-findings.md                 # THE GATE: method-selection decision (deliverable)
data/                                    # gitignored audio + artifacts
```

---

### Task 1: Project scaffold, config, and smoke test

**Files:**
- Create: `pyproject.toml`, `README.md`, `src/mol/__init__.py`, `src/mol/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `mol.config.SEED_LANGUAGES` (`dict[str, LanguageSpec]`), `mol.config.DATA_DIR` (`pathlib.Path`), `mol.config.LanguageSpec` (dataclass with `name: str`, `iso639_1: str`, `radio_browser_lang: str`).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "mol"
version = "0.0.1"
description = "The Music of Languages — prosody-based language proximity (Phase 0 exploration)"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "praat-parselmouth>=0.4.3",
    "librosa>=0.10",
    "soundfile>=0.12",
    "silero-vad>=5.1",
    "torch>=2.2",
    "requests>=2.31",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.12", "jupyter>=1.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/mol/__init__.py`**

```python
"""The Music of Languages — Phase 0 exploration package."""

__version__ = "0.0.1"
```

- [ ] **Step 3: Write the failing test for config**

`tests/test_config.py`:

```python
from mol import config


def test_eight_seed_languages_present():
    assert len(config.SEED_LANGUAGES) == 8
    assert set(config.SEED_LANGUAGES) == {
        "english", "german", "polish", "french",
        "spanish", "italian", "greek", "finnish",
    }


def test_language_spec_fields():
    english = config.SEED_LANGUAGES["english"]
    assert english.name == "English"
    assert english.iso639_1 == "en"
    assert english.radio_browser_lang == "english"


def test_data_dir_is_path():
    from pathlib import Path
    assert isinstance(config.DATA_DIR, Path)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.config'` (or ImportError).

- [ ] **Step 5: Implement `src/mol/config.py`**

```python
"""Project-wide configuration: seed languages and paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LanguageSpec:
    name: str               # display name
    iso639_1: str           # 2-letter code
    radio_browser_lang: str # 'language' value used by radio-browser.info


SEED_LANGUAGES: dict[str, LanguageSpec] = {
    "english": LanguageSpec("English", "en", "english"),
    "german":  LanguageSpec("German",  "de", "german"),
    "polish":  LanguageSpec("Polish",  "pl", "polish"),
    "french":  LanguageSpec("French",  "fr", "french"),
    "spanish": LanguageSpec("Spanish", "es", "spanish"),
    "italian": LanguageSpec("Italian", "it", "italian"),
    "greek":   LanguageSpec("Greek",   "el", "greek"),
    "finnish": LanguageSpec("Finnish", "fi", "finnish"),
}

# Repo root is two parents up from this file (src/mol/config.py -> repo root).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"

# Audio working format used throughout the pipeline.
TARGET_SAMPLE_RATE: int = 16_000
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]" && .venv/Scripts/python -m pytest tests/test_config.py -v`
(On macOS/Linux use `.venv/bin/python`.)
Expected: 3 passed.

- [ ] **Step 7: Write `README.md`**

```markdown
# The Music of Languages — Phase 0

Exploration & method-selection for prosody-based language proximity.
See `docs/superpowers/specs/2026-06-24-music-of-languages-design.md` for the design.

## Setup (Windows PowerShell)

    python -m venv .venv
    .venv\Scripts\python -m pip install -e ".[dev]"

## Run tests

    .venv\Scripts\python -m pytest -v

## Notebooks

    .venv\Scripts\jupyter notebook notebooks/
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md src/mol/__init__.py src/mol/config.py tests/test_config.py
git commit -m "feat: scaffold mol package with config and smoke test"
```

---

### Task 2: Audio loading & loudness normalization

**Files:**
- Create: `src/mol/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `mol.config.TARGET_SAMPLE_RATE`.
- Produces:
  - `load_audio(path: str | Path, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray` — mono float32 in [-1, 1].
  - `normalize_loudness(signal: np.ndarray, target_rms: float = 0.1) -> np.ndarray` — RMS-scaled copy; returns input unchanged if silent.

- [ ] **Step 1: Write the failing test**

`tests/test_audio.py`:

```python
import numpy as np
import soundfile as sf

from mol import audio


def test_normalize_loudness_sets_target_rms():
    sig = np.random.default_rng(0).normal(0, 0.5, 16_000).astype(np.float32)
    out = audio.normalize_loudness(sig, target_rms=0.1)
    rms = float(np.sqrt(np.mean(out**2)))
    assert abs(rms - 0.1) < 1e-3


def test_normalize_loudness_silent_signal_is_safe():
    sig = np.zeros(1000, dtype=np.float32)
    out = audio.normalize_loudness(sig, target_rms=0.1)
    assert np.allclose(out, 0.0)
    assert not np.any(np.isnan(out))


def test_load_audio_returns_mono_target_sr(tmp_path):
    # Write a 2-channel 8 kHz wav; expect mono 16 kHz back.
    sr_in = 8_000
    t = np.linspace(0, 1.0, sr_in, endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([tone, tone], axis=1).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(p, stereo, sr_in)

    out = audio.load_audio(p, sr=16_000)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert abs(len(out) - 16_000) <= 2  # resampled to ~1 s at 16 kHz
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.audio'`.

- [ ] **Step 3: Implement `src/mol/audio.py`**

```python
"""Audio loading and loudness normalization."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from mol.config import TARGET_SAMPLE_RATE


def load_audio(path: str | Path, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at `sr` Hz, range [-1, 1]."""
    signal, _ = librosa.load(str(path), sr=sr, mono=True)
    return signal.astype(np.float32)


def normalize_loudness(signal: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale `signal` to `target_rms`. Silent signals are returned unchanged."""
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return signal.astype(np.float32)
    return (signal * (target_rms / rms)).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_audio.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/audio.py tests/test_audio.py
git commit -m "feat: add audio loading and loudness normalization"
```

---

### Task 3: VAD speech extraction (silero-vad)

**Files:**
- Create: `src/mol/clean/__init__.py`, `src/mol/clean/vad.py`
- Test: `tests/test_vad.py`

**Interfaces:**
- Consumes: `mol.config.TARGET_SAMPLE_RATE`.
- Produces:
  - `merge_segments(segments: list[tuple[float, float]], gap: float = 0.2) -> list[tuple[float, float]]` — pure: merges (start, end) seconds closer than `gap`.
  - `total_speech_seconds(segments: list[tuple[float, float]]) -> float` — pure.
  - `extract_speech(signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, min_speech_s: float = 0.5) -> list[tuple[float, float]]` — runs silero-vad, returns merged speech segments. (Integration; the model is loaded lazily and cached.)
  - `concat_speech(signal, segments, sr=TARGET_SAMPLE_RATE) -> np.ndarray` — pure: concatenate the speech spans into one array.

The pure helpers are the unit-tested core; `extract_speech` is exercised in the notebook (integration).

- [ ] **Step 1: Write the failing test (pure helpers only)**

`tests/test_vad.py`:

```python
import numpy as np

from mol.clean import vad


def test_merge_segments_joins_close_spans():
    segs = [(0.0, 1.0), (1.1, 2.0), (5.0, 6.0)]
    merged = vad.merge_segments(segs, gap=0.2)
    assert merged == [(0.0, 2.0), (5.0, 6.0)]


def test_merge_segments_keeps_distant_spans():
    segs = [(0.0, 1.0), (3.0, 4.0)]
    assert vad.merge_segments(segs, gap=0.2) == [(0.0, 1.0), (3.0, 4.0)]


def test_merge_segments_empty():
    assert vad.merge_segments([], gap=0.2) == []


def test_total_speech_seconds():
    assert vad.total_speech_seconds([(0.0, 1.0), (2.0, 3.5)]) == 2.5


def test_concat_speech_picks_correct_samples():
    sr = 1000
    signal = np.arange(3000, dtype=np.float32)  # 3 s at 1 kHz
    out = vad.concat_speech(signal, [(0.0, 1.0), (2.0, 3.0)], sr=sr)
    expected = np.concatenate([np.arange(0, 1000), np.arange(2000, 3000)]).astype(np.float32)
    assert np.array_equal(out, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_vad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.clean.vad'`.

- [ ] **Step 3: Implement `src/mol/clean/__init__.py` (empty) and `src/mol/clean/vad.py`**

`src/mol/clean/__init__.py`:

```python
```

`src/mol/clean/vad.py`:

```python
"""Voice-activity detection: extract speech spans with silero-vad.

Pure span helpers are unit-tested; `extract_speech` wraps the model (integration).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from mol.config import TARGET_SAMPLE_RATE


def merge_segments(
    segments: list[tuple[float, float]], gap: float = 0.2
) -> list[tuple[float, float]]:
    """Merge (start, end) spans (seconds) separated by less than `gap`."""
    if not segments:
        return []
    ordered = sorted(segments)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def total_speech_seconds(segments: list[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in segments))


def concat_speech(
    signal: np.ndarray, segments: list[tuple[float, float]], sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Concatenate the samples inside `segments` into a single array."""
    parts = [signal[int(start * sr):int(end * sr)] for start, end in segments]
    if not parts:
        return np.zeros(0, dtype=signal.dtype)
    return np.concatenate(parts).astype(signal.dtype)


@lru_cache(maxsize=1)
def _load_model():
    # Imported lazily so unit tests of the pure helpers don't need torch.
    from silero_vad import load_silero_vad

    return load_silero_vad()


def extract_speech(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, min_speech_s: float = 0.5
) -> list[tuple[float, float]]:
    """Return merged speech spans (seconds) detected by silero-vad."""
    import torch
    from silero_vad import get_speech_timestamps

    model = _load_model()
    tensor = torch.from_numpy(np.ascontiguousarray(signal))
    stamps = get_speech_timestamps(
        tensor, model, sampling_rate=sr, return_seconds=True,
        min_speech_duration_ms=int(min_speech_s * 1000),
    )
    segs = [(float(s["start"]), float(s["end"])) for s in stamps]
    return merge_segments(segs, gap=0.2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_vad.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/clean/__init__.py src/mol/clean/vad.py tests/test_vad.py
git commit -m "feat: add VAD speech extraction with tested span helpers"
```

---

### Task 4: Radio ingest helper (radio-browser.info + ffmpeg)

**Files:**
- Create: `src/mol/ingest/__init__.py`, `src/mol/ingest/radio.py`
- Test: `tests/test_radio.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (standalone helper).
- Produces:
  - `find_stations(language: str, limit: int = 10, tags: str = "talk,news", session=None) -> list[Station]` where `Station` is a dataclass `(name: str, url: str, codec: str, bitrate: int)`.
  - `record_clip(stream_url: str, out_path: str | Path, duration_s: int = 60, runner=subprocess.run) -> Path` — invokes ffmpeg to capture `duration_s` seconds to `out_path`. `runner` is injectable for testing.

Network (`requests`) and subprocess (`ffmpeg`) are mocked in unit tests.

- [ ] **Step 1: Write the failing test**

`tests/test_radio.py`:

```python
from pathlib import Path

from mol.ingest import radio


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        return _FakeResponse(self._payload)


def test_find_stations_parses_payload_and_filters_blank_urls():
    payload = [
        {"name": "Talk One", "url_resolved": "http://a", "codec": "MP3", "bitrate": 128},
        {"name": "No URL", "url_resolved": "", "codec": "MP3", "bitrate": 128},
        {"name": "Talk Two", "url_resolved": "http://b", "codec": "AAC", "bitrate": 64},
    ]
    session = _FakeSession(payload)
    stations = radio.find_stations("german", limit=10, session=session)
    assert [s.name for s in stations] == ["Talk One", "Talk Two"]
    assert stations[0].url == "http://a"
    # language must be passed to the API
    _, params = session.calls[0]
    assert params["language"] == "german"


def test_record_clip_builds_ffmpeg_command(tmp_path):
    captured = {}

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"RIFFfake")
        class R:  # noqa: D401
            returncode = 0
        return R()

    out = tmp_path / "clip.wav"
    result = radio.record_clip("http://a", out, duration_s=30, runner=fake_runner)
    assert result == out
    assert out.exists()
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "30" in cmd            # duration appears
    assert "http://a" in cmd      # input url appears
    assert str(out) == cmd[-1]    # output path is last
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_radio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.ingest.radio'`.

- [ ] **Step 3: Implement `src/mol/ingest/__init__.py` (empty) and `src/mol/ingest/radio.py`**

`src/mol/ingest/__init__.py`:

```python
```

`src/mol/ingest/radio.py`:

```python
"""Lightweight, portable radio ingest helper (Phase 0 prototype).

Phase 1 hardens this (multi-station/time sampling, retries, scaling).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

# A public radio-browser.info mirror. Phase 1 should resolve a mirror dynamically.
API_BASE = "https://de1.api.radio-browser.info"
USER_AGENT = "music-of-languages/0.0 (research; contact: project owner)"


@dataclass(frozen=True)
class Station:
    name: str
    url: str
    codec: str
    bitrate: int


def find_stations(
    language: str, limit: int = 10, tags: str = "talk,news", session=None
) -> list[Station]:
    """Return up to `limit` stations for `language`, biased to talk/news tags."""
    session = session or requests.Session()
    params = {
        "language": language,
        "tagList": tags,
        "limit": limit,
        "hidebroken": "true",
        "order": "votes",
        "reverse": "true",
    }
    resp = session.get(
        f"{API_BASE}/json/stations/search",
        params=params,
        timeout=20,
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    stations: list[Station] = []
    for row in resp.json():
        url = row.get("url_resolved") or row.get("url") or ""
        if not url:
            continue
        stations.append(
            Station(
                name=row.get("name", "").strip(),
                url=url,
                codec=row.get("codec", ""),
                bitrate=int(row.get("bitrate", 0) or 0),
            )
        )
    return stations


def record_clip(
    stream_url: str,
    out_path: str | Path,
    duration_s: int = 60,
    runner=subprocess.run,
) -> Path:
    """Capture `duration_s` seconds of `stream_url` to `out_path` (wav) via ffmpeg."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-t", str(duration_s),
        "-i", stream_url,
        "-ac", "1", "-ar", "16000",
        str(out_path),
    ]
    runner(cmd, check=True)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_radio.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/ingest/__init__.py src/mol/ingest/radio.py tests/test_radio.py
git commit -m "feat: add portable radio ingest helper (stations + ffmpeg capture)"
```

---

### Task 5: FeatureExtractor interface

**Files:**
- Create: `src/mol/features/__init__.py`, `src/mol/features/base.py`
- Test: `tests/test_features_base.py`

**Interfaces:**
- Produces:
  - `FeatureVector = dict[str, float]` (type alias).
  - `FeatureExtractor` ABC with `name: str` (property) and `extract(self, signal: np.ndarray, sr: int) -> FeatureVector`.
  - A concrete `ConstantExtractor(value: float)` used only to prove the interface (kept in `base.py` for tests/examples).

- [ ] **Step 1: Write the failing test**

`tests/test_features_base.py`:

```python
import numpy as np
import pytest

from mol.features.base import FeatureExtractor, ConstantExtractor


def test_constant_extractor_implements_interface():
    ex = ConstantExtractor(value=1.5)
    assert isinstance(ex, FeatureExtractor)
    assert ex.name == "constant"
    out = ex.extract(np.zeros(100, dtype=np.float32), sr=16_000)
    assert out == {"constant": 1.5}


def test_feature_extractor_is_abstract():
    with pytest.raises(TypeError):
        FeatureExtractor()  # cannot instantiate ABC
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_features_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.features.base'`.

- [ ] **Step 3: Implement `src/mol/features/__init__.py` (empty) and `src/mol/features/base.py`**

`src/mol/features/__init__.py`:

```python
```

`src/mol/features/base.py`:

```python
"""The pluggable feature-extraction interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

FeatureVector = dict[str, float]


class FeatureExtractor(ABC):
    """Maps a single speech clip to a flat, named feature vector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used to namespace this extractor's outputs."""

    @abstractmethod
    def extract(self, signal: np.ndarray, sr: int) -> FeatureVector:
        """Return a {feature_name: value} dict for one clip."""


class ConstantExtractor(FeatureExtractor):
    """Trivial extractor used to validate the interface."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    @property
    def name(self) -> str:
        return "constant"

    def extract(self, signal: np.ndarray, sr: int) -> FeatureVector:
        return {"constant": float(self._value)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_features_base.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/features/__init__.py src/mol/features/base.py tests/test_features_base.py
git commit -m "feat: add pluggable FeatureExtractor interface"
```

---

### Task 6: Rhythm-metric math (pure functions)

**Files:**
- Create: `src/mol/features/rhythm_metrics.py`
- Test: `tests/test_rhythm_metrics.py`

This is the most literature-faithful, fully deterministic unit. Formulas per Ramus et al. (1999), Grabe & Low (2002), Dellwo (2006).

**Interfaces:**
- Produces (all take `intervals: list[float]` durations, in seconds, and return `float`):
  - `percent_v(vocalic: list[float], consonantal: list[float]) -> float` — 100 * sumV / (sumV + sumC).
  - `delta(intervals) -> float` — population SD of durations.
  - `varco(intervals) -> float` — 100 * SD / mean (coefficient of variation).
  - `npvi(intervals) -> float` — normalized Pairwise Variability Index.
  - `rpvi(intervals) -> float` — raw PVI.

- [ ] **Step 1: Write the failing test (exact expected values)**

`tests/test_rhythm_metrics.py`:

```python
import math

import pytest

from mol.features import rhythm_metrics as rm


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_rhythm_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.features.rhythm_metrics'`.

- [ ] **Step 3: Implement `src/mol/features/rhythm_metrics.py`**

```python
"""Duration-based rhythm metrics (pure functions).

References: Ramus, Nespor & Mehler (1999); Grabe & Low (2002); Dellwo (2006).
All inputs are lists of interval durations in seconds. NaN is returned where a
metric is undefined (too few intervals / empty input).
"""

from __future__ import annotations

import math
import statistics


def percent_v(vocalic: list[float], consonantal: list[float]) -> float:
    """Percentage of total interval duration that is vocalic."""
    total = sum(vocalic) + sum(consonantal)
    if total <= 0:
        return math.nan
    return 100.0 * sum(vocalic) / total


def delta(intervals: list[float]) -> float:
    """Population standard deviation of interval durations (ΔC / ΔV)."""
    if len(intervals) < 2:
        return math.nan
    return statistics.pstdev(intervals)


def varco(intervals: list[float]) -> float:
    """Rate-normalized delta: 100 * SD / mean (coefficient of variation)."""
    if len(intervals) < 2:
        return math.nan
    mean = statistics.fmean(intervals)
    if mean <= 0:
        return math.nan
    return 100.0 * statistics.pstdev(intervals) / mean


def npvi(intervals: list[float]) -> float:
    """Normalized Pairwise Variability Index (rate-normalized)."""
    if len(intervals) < 2:
        return math.nan
    pairs = zip(intervals[:-1], intervals[1:])
    terms = [abs(a - b) / ((a + b) / 2.0) for a, b in pairs if (a + b) > 0]
    if not terms:
        return math.nan
    return 100.0 * statistics.fmean(terms)


def rpvi(intervals: list[float]) -> float:
    """Raw Pairwise Variability Index (not rate-normalized)."""
    if len(intervals) < 2:
        return math.nan
    diffs = [abs(a - b) for a, b in zip(intervals[:-1], intervals[1:])]
    return statistics.fmean(diffs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_rhythm_metrics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/features/rhythm_metrics.py tests/test_rhythm_metrics.py
git commit -m "feat: add duration-based rhythm metrics (pure, literature-faithful)"
```

---

### Task 7: Automatic vocalic/consonantal interval detection

**Files:**
- Create: `src/mol/features/intervals.py`
- Test: `tests/test_intervals.py`

Alignment-free, approximate C/V segmentation. A frame is "vocalic" if it is voiced (F0 present) AND high-intensity relative to the clip; contiguous vocalic frames form vocalic intervals, the gaps between them (within speech) form consonantal intervals. This is the documented Phase-0 approximation the exploration evaluates (research: %V-style measures survive automatic segmentation).

**Interfaces:**
- Consumes: `mol.config.TARGET_SAMPLE_RATE`.
- Produces:
  - `frames_to_intervals(is_vocalic: list[bool], frame_step: float) -> tuple[list[float], list[float]]` — pure: returns (vocalic_intervals, consonantal_intervals) in seconds, where consonantal intervals are non-vocalic runs strictly *between* two vocalic runs.
  - `detect_intervals(signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> tuple[list[float], list[float]]` — uses parselmouth pitch + intensity to produce the boolean frame mask, then calls `frames_to_intervals`. (Integration; thin wrapper.)

- [ ] **Step 1: Write the failing test (pure core)**

`tests/test_intervals.py`:

```python
import pytest

from mol.features import intervals


def test_frames_to_intervals_basic():
    # V C V  pattern: 2 vocalic frames, 1 consonantal, 2 vocalic; step 0.01 s
    mask = [True, True, False, True, True]
    voc, cons = intervals.frames_to_intervals(mask, frame_step=0.01)
    assert voc == pytest.approx([0.02, 0.02])
    assert cons == pytest.approx([0.01])


def test_leading_and_trailing_nonvocalic_are_not_consonantal():
    # Only gaps BETWEEN vocalic runs count as consonantal intervals.
    mask = [False, True, False, False, True, False]
    voc, cons = intervals.frames_to_intervals(mask, frame_step=0.01)
    assert voc == pytest.approx([0.01, 0.01])
    assert cons == pytest.approx([0.02])


def test_all_silence_yields_nothing():
    voc, cons = intervals.frames_to_intervals([False, False], frame_step=0.01)
    assert voc == []
    assert cons == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_intervals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.features.intervals'`.

- [ ] **Step 3: Implement `src/mol/features/intervals.py`**

```python
"""Alignment-free vocalic/consonantal interval detection (Phase 0 approximation)."""

from __future__ import annotations

import numpy as np
import parselmouth

from mol.config import TARGET_SAMPLE_RATE

_FRAME_STEP = 0.01      # 10 ms analysis step
_INTENSITY_PERCENTILE = 40  # frames below this percentile are treated as non-vocalic


def frames_to_intervals(
    is_vocalic: list[bool], frame_step: float
) -> tuple[list[float], list[float]]:
    """Convert a per-frame vocalic mask into vocalic and (medial) consonantal durations."""
    # Run-length encode the boolean mask.
    runs: list[tuple[bool, int]] = []
    for value in is_vocalic:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))

    vocalic = [length * frame_step for value, length in runs if value]

    # Consonantal = non-vocalic runs that sit strictly between two vocalic runs.
    consonantal: list[float] = []
    for i, (value, length) in enumerate(runs):
        if not value and 0 < i < len(runs) - 1:
            consonantal.append(length * frame_step)
    return vocalic, consonantal


def detect_intervals(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE
) -> tuple[list[float], list[float]]:
    """Detect vocalic/consonantal intervals via voicing + relative intensity."""
    sound = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=_FRAME_STEP)
    intensity = sound.to_intensity(time_step=_FRAME_STEP)

    f0 = pitch.selected_array["frequency"]  # 0.0 where unvoiced
    times = pitch.xs()
    intens = np.array([intensity.get_value(t) or 0.0 for t in times])
    voiced = f0 > 0
    if voiced.any():
        threshold = np.percentile(intens[voiced], _INTENSITY_PERCENTILE)
    else:
        threshold = np.inf
    is_vocalic = [(bool(v) and i >= threshold) for v, i in zip(voiced, intens)]
    return frames_to_intervals(is_vocalic, frame_step=_FRAME_STEP)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_intervals.py -v`
Expected: 3 passed.

- [ ] **Step 5: Add an integration sanity check (skipped if no audio fixture)**

Append to `tests/test_intervals.py`:

```python
import numpy as np


def test_detect_intervals_on_synthetic_voiced_tone_runs():
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 150 Hz tone (voiced) with a silent gap in the middle.
    tone = 0.3 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
    tone[sr // 2 - 500: sr // 2 + 500] = 0.0
    voc, cons = intervals.detect_intervals(tone, sr=sr)
    assert sum(voc) > 0.0  # detected some vocalic content
```

- [ ] **Step 6: Run and commit**

Run: `.venv/Scripts/python -m pytest tests/test_intervals.py -v`
Expected: 4 passed.

```bash
git add src/mol/features/intervals.py tests/test_intervals.py
git commit -m "feat: add alignment-free vocalic/consonantal interval detection"
```

---

### Task 8: Pitch / intonation features

**Files:**
- Create: `src/mol/features/pitch.py`
- Test: `tests/test_pitch.py`

**Interfaces:**
- Consumes: `mol.config.TARGET_SAMPLE_RATE`.
- Produces: `pitch_features(signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> dict[str, float]` with keys
  `f0_mean, f0_std, f0_min, f0_max, f0_range, f0_slope` (Hz; computed over voiced frames; NaN-safe).

- [ ] **Step 1: Write the failing test**

`tests/test_pitch.py`:

```python
import numpy as np

from mol.features import pitch


def test_pitch_features_constant_tone():
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    feats = pitch.pitch_features(tone, sr=sr)
    assert set(feats) == {
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope",
    }
    # Parselmouth should recover ~200 Hz for a clean 200 Hz tone.
    assert abs(feats["f0_mean"] - 200.0) < 10.0
    assert feats["f0_std"] < 5.0


def test_pitch_features_silence_is_nan_safe():
    feats = pitch.pitch_features(np.zeros(16_000, dtype=np.float32), sr=16_000)
    assert all(isinstance(v, float) for v in feats.values())
    assert np.isnan(feats["f0_mean"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_pitch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.features.pitch'`.

- [ ] **Step 3: Implement `src/mol/features/pitch.py`**

```python
"""F0 / intonation features via parselmouth."""

from __future__ import annotations

import math

import numpy as np
import parselmouth

from mol.config import TARGET_SAMPLE_RATE


def pitch_features(signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> dict[str, float]:
    """Summary statistics of the F0 contour over voiced frames."""
    keys = ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope"]
    sound = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.01)
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    voiced = f0 > 0
    if voiced.sum() < 2:
        return {k: math.nan for k in keys}

    fv = f0[voiced]
    tv = times[voiced]
    slope = float(np.polyfit(tv - tv[0], fv, 1)[0])  # Hz/s linear trend
    return {
        "f0_mean": float(np.mean(fv)),
        "f0_std": float(np.std(fv)),
        "f0_min": float(np.min(fv)),
        "f0_max": float(np.max(fv)),
        "f0_range": float(np.max(fv) - np.min(fv)),
        "f0_slope": slope,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_pitch.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/features/pitch.py tests/test_pitch.py
git commit -m "feat: add F0/intonation features"
```

---

### Task 9: Speech-rate via syllable nuclei

**Files:**
- Create: `src/mol/features/speech_rate.py`
- Test: `tests/test_speech_rate.py`

Syllable-nuclei detection in the De Jong & Wempe (2009) spirit: intensity peaks that exceed neighbouring dips by a threshold and are voiced. The peak-picking logic is a pure function over an intensity contour and is unit-tested; the parselmouth wrapper is a thin integration layer.

**Interfaces:**
- Consumes: `mol.config.TARGET_SAMPLE_RATE`.
- Produces:
  - `count_nuclei(intensity_db: np.ndarray, voiced: np.ndarray, min_dip_db: float = 2.0) -> int` — pure.
  - `speech_rate_features(signal, sr=TARGET_SAMPLE_RATE) -> dict[str, float]` with keys
    `syllables_per_sec, n_syllables, duration_s`. (Integration.)

- [ ] **Step 1: Write the failing test (pure peak-picker)**

`tests/test_speech_rate.py`:

```python
import numpy as np

from mol.features import speech_rate


def test_count_nuclei_three_clear_peaks():
    # Three peaks separated by deep dips, all voiced.
    db = np.array([40, 60, 40, 62, 41, 58, 40], dtype=float)
    voiced = np.ones_like(db, dtype=bool)
    assert speech_rate.count_nuclei(db, voiced, min_dip_db=2.0) == 3


def test_count_nuclei_ignores_unvoiced_peaks():
    db = np.array([40, 60, 40, 62, 40], dtype=float)
    voiced = np.array([1, 0, 1, 1, 1], dtype=bool)  # first peak unvoiced
    assert speech_rate.count_nuclei(db, voiced, min_dip_db=2.0) == 1


def test_count_nuclei_requires_min_dip():
    # Tiny ripples (dip < threshold) should not count as separate nuclei.
    db = np.array([40, 60, 59, 60, 40], dtype=float)
    voiced = np.ones_like(db, dtype=bool)
    assert speech_rate.count_nuclei(db, voiced, min_dip_db=2.0) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_speech_rate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.features.speech_rate'`.

- [ ] **Step 3: Implement `src/mol/features/speech_rate.py`**

```python
"""Speech-rate estimation via syllable-nuclei detection (De Jong & Wempe style)."""

from __future__ import annotations

import math

import numpy as np
import parselmouth
from scipy.signal import find_peaks

from mol.config import TARGET_SAMPLE_RATE


def count_nuclei(
    intensity_db: np.ndarray, voiced: np.ndarray, min_dip_db: float = 2.0
) -> int:
    """Count intensity peaks that clear `min_dip_db` prominence and are voiced."""
    peaks, _ = find_peaks(intensity_db, prominence=min_dip_db)
    return int(sum(1 for p in peaks if voiced[p]))


def speech_rate_features(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE
) -> dict[str, float]:
    """Estimate syllable rate from intensity peaks filtered by voicing."""
    duration = len(signal) / sr
    if duration <= 0:
        return {"syllables_per_sec": math.nan, "n_syllables": 0.0, "duration_s": 0.0}

    sound = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=sr)
    intensity = sound.to_intensity(time_step=0.01)
    pitch = sound.to_pitch(time_step=0.01)

    db = np.array(intensity.values[0])
    times = intensity.xs()
    voiced = np.array(
        [(pitch.get_value_at_time(t) or 0.0) > 0 for t in times], dtype=bool
    )
    n = count_nuclei(db, voiced, min_dip_db=2.0)
    return {
        "syllables_per_sec": float(n / duration),
        "n_syllables": float(n),
        "duration_s": float(duration),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_speech_rate.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/features/speech_rate.py tests/test_speech_rate.py
git commit -m "feat: add syllable-nuclei speech-rate estimation"
```

---

### Task 10: Prosody-acoustic extractor (assembles the pieces)

**Files:**
- Create: `src/mol/features/prosody_acoustic.py`
- Test: `tests/test_prosody_acoustic.py`

**Interfaces:**
- Consumes: `FeatureExtractor`, `pitch.pitch_features`, `speech_rate.speech_rate_features`, `intervals.detect_intervals`, `rhythm_metrics.*`.
- Produces: `ProsodyAcousticExtractor()` implementing `FeatureExtractor`; `name == "prosody_acoustic"`; `extract` returns a `FeatureVector` whose keys are the union of:
  `f0_mean, f0_std, f0_min, f0_max, f0_range, f0_slope, syllables_per_sec, n_syllables, duration_s, percent_v, delta_v, delta_c, varco_v, varco_c, npvi_v, rpvi_c`.

- [ ] **Step 1: Write the failing test**

`tests/test_prosody_acoustic.py`:

```python
import numpy as np

from mol.features.base import FeatureExtractor
from mol.features.prosody_acoustic import ProsodyAcousticExtractor

EXPECTED_KEYS = {
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope",
    "syllables_per_sec", "n_syllables", "duration_s",
    "percent_v", "delta_v", "delta_c", "varco_v", "varco_c", "npvi_v", "rpvi_c",
}


def test_extractor_is_a_feature_extractor():
    ex = ProsodyAcousticExtractor()
    assert isinstance(ex, FeatureExtractor)
    assert ex.name == "prosody_acoustic"


def test_extract_returns_all_keys_as_floats():
    sr = 16_000
    rng = np.random.default_rng(1)
    # 2 s of amplitude-modulated voiced tone -> exercises pitch/rate/intervals.
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))  # 4 Hz syllable-like modulation
    sig = (env * 0.3 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
    sig += rng.normal(0, 0.001, sig.shape).astype(np.float32)

    feats = ProsodyAcousticExtractor().extract(sig, sr=sr)
    assert set(feats) == EXPECTED_KEYS
    assert all(isinstance(v, float) for v in feats.values())
    assert feats["duration_s"] > 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_prosody_acoustic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.features.prosody_acoustic'`.

- [ ] **Step 3: Implement `src/mol/features/prosody_acoustic.py`**

```python
"""The alignment-free prosody/rhythm FeatureExtractor (Phase 0 primary candidate).

Inspired by the DisVoice prosody feature set, but self-contained (parselmouth +
our own rhythm metrics) so features stay tweakable during exploration.
"""

from __future__ import annotations

import numpy as np

from mol.config import TARGET_SAMPLE_RATE
from mol.features import intervals, pitch, rhythm_metrics, speech_rate
from mol.features.base import FeatureExtractor, FeatureVector


class ProsodyAcousticExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "prosody_acoustic"

    def extract(self, signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> FeatureVector:
        feats: FeatureVector = {}
        feats.update(pitch.pitch_features(signal, sr=sr))
        feats.update(speech_rate.speech_rate_features(signal, sr=sr))

        voc, cons = intervals.detect_intervals(signal, sr=sr)
        feats["percent_v"] = rhythm_metrics.percent_v(voc, cons)
        feats["delta_v"] = rhythm_metrics.delta(voc)
        feats["delta_c"] = rhythm_metrics.delta(cons)
        feats["varco_v"] = rhythm_metrics.varco(voc)
        feats["varco_c"] = rhythm_metrics.varco(cons)
        feats["npvi_v"] = rhythm_metrics.npvi(voc)
        feats["rpvi_c"] = rhythm_metrics.rpvi(cons)
        return {k: float(v) for k, v in feats.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_prosody_acoustic.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/features/prosody_acoustic.py tests/test_prosody_acoustic.py
git commit -m "feat: assemble alignment-free prosody-acoustic extractor"
```

---

### Task 11: Per-language aggregation

**Files:**
- Create: `src/mol/features/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `FeatureVector` (dict[str, float]).
- Produces:
  - `aggregate_language(clip_vectors: list[FeatureVector]) -> dict[str, float]` — per-feature `mean` and `std` (suffixed `_mean`/`_std`), ignoring NaNs.
  - `build_language_table(per_language: dict[str, list[FeatureVector]]) -> pandas.DataFrame` — index = language, columns = aggregated features.

- [ ] **Step 1: Write the failing test**

`tests/test_aggregate.py`:

```python
import math

import numpy as np
import pandas as pd

from mol.features import aggregate


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.features.aggregate'`.

- [ ] **Step 3: Implement `src/mol/features/aggregate.py`**

```python
"""Aggregate per-clip feature vectors into per-language rows (mean + dispersion)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_language(clip_vectors: list[dict[str, float]]) -> dict[str, float]:
    """Per-feature mean and std across clips, ignoring NaNs."""
    keys = sorted({k for v in clip_vectors for k in v})
    out: dict[str, float] = {}
    for k in keys:
        values = np.array([v.get(k, np.nan) for v in clip_vectors], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            out[f"{k}_mean"] = np.nan
            out[f"{k}_std"] = np.nan
        else:
            out[f"{k}_mean"] = float(np.mean(values))
            out[f"{k}_std"] = float(np.std(values))
    return out


def build_language_table(
    per_language: dict[str, list[dict[str, float]]]
) -> pd.DataFrame:
    """Build a language-indexed DataFrame of aggregated features."""
    rows = {lang: aggregate_language(vecs) for lang, vecs in per_language.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_aggregate.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/features/aggregate.py tests/test_aggregate.py
git commit -m "feat: add per-language feature aggregation"
```

---

### Task 12: Proximity utilities (distance matrix, clustering, MDS)

**Files:**
- Create: `src/mol/proximity/__init__.py`, `src/mol/proximity/distance.py`
- Test: `tests/test_distance.py`

**Interfaces:**
- Consumes: a feature DataFrame (language-indexed, numeric columns).
- Produces:
  - `standardize(df) -> pandas.DataFrame` — z-score each column (NaN columns dropped).
  - `distance_matrix(df, metric="euclidean") -> pandas.DataFrame` — square, symmetric, language×language.
  - `linkage_matrix(dist_df, method="ward") -> np.ndarray` — scipy linkage for dendrograms.
  - `mds_2d(dist_df, seed=0) -> pandas.DataFrame` — 2 columns (`mds_x`, `mds_y`), language-indexed.

- [ ] **Step 1: Write the failing test**

`tests/test_distance.py`:

```python
import numpy as np
import pandas as pd

from mol.proximity import distance


def _toy():
    return pd.DataFrame(
        {"a": [0.0, 1.0, 10.0], "b": [0.0, 1.0, 10.0]},
        index=["x", "y", "z"],
    )


def test_standardize_zero_mean_unit_std():
    out = distance.standardize(_toy())
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-9)


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    dm = distance.distance_matrix(distance.standardize(_toy()))
    assert list(dm.index) == ["x", "y", "z"]
    assert np.allclose(np.diag(dm.values), 0.0)
    assert np.allclose(dm.values, dm.values.T)
    # x and y are closer to each other than to z
    assert dm.loc["x", "y"] < dm.loc["x", "z"]


def test_mds_2d_shape():
    dm = distance.distance_matrix(distance.standardize(_toy()))
    coords = distance.mds_2d(dm, seed=0)
    assert list(coords.columns) == ["mds_x", "mds_y"]
    assert list(coords.index) == ["x", "y", "z"]


def test_linkage_matrix_rows():
    dm = distance.distance_matrix(distance.standardize(_toy()))
    Z = distance.linkage_matrix(dm)
    assert Z.shape == (2, 4)  # n-1 merges for 3 leaves
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_distance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.proximity.distance'`.

- [ ] **Step 3: Implement `src/mol/proximity/__init__.py` (empty) and `src/mol/proximity/distance.py`**

`src/mol/proximity/__init__.py`:

```python
```

`src/mol/proximity/distance.py`:

```python
"""Language proximity: standardization, distances, clustering, MDS embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each numeric column; drop columns that are all-NaN or constant."""
    numeric = df.select_dtypes("number").dropna(axis=1, how="any")
    std = numeric.std(axis=0, ddof=0)
    numeric = numeric.loc[:, std > 0]
    return (numeric - numeric.mean(axis=0)) / numeric.std(axis=0, ddof=0)


def distance_matrix(df: pd.DataFrame, metric: str = "euclidean") -> pd.DataFrame:
    """Square, symmetric language×language distance matrix."""
    condensed = pdist(df.values, metric=metric)
    square = squareform(condensed)
    return pd.DataFrame(square, index=df.index, columns=df.index)


def linkage_matrix(dist_df: pd.DataFrame, method: str = "ward") -> np.ndarray:
    """SciPy linkage matrix for dendrograms, from a square distance frame."""
    condensed = squareform(dist_df.values, checks=False)
    return linkage(condensed, method=method)


def mds_2d(dist_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """2-D metric MDS embedding from a precomputed distance matrix."""
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, normalized_stress=False)
    coords = mds.fit_transform(dist_df.values)
    return pd.DataFrame(coords, index=dist_df.index, columns=["mds_x", "mds_y"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_distance.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mol/proximity/__init__.py src/mol/proximity/distance.py tests/test_distance.py
git commit -m "feat: add proximity utilities (standardize, distance, linkage, MDS)"
```

---

### Task 13: Typology reference & validation

**Files:**
- Create: `src/mol/validation/__init__.py`, `src/mol/validation/typology.py`
- Test: `tests/test_typology.py`

Reference rhythm-class labels for the seed languages, with verified nPVI values where available (Grabe & Low 2002). Validation asks: do stress-timed languages get higher computed vocalic nPVI than syllable-timed ones? Labels are intentionally conservative (Greek/Finnish marked `intermediate`) and meant to be revisited in exploration.

**Interfaces:**
- Produces:
  - `RHYTHM_CLASS: dict[str, str]` — language -> {"stress", "syllable", "intermediate"}.
  - `REFERENCE_NPVI_V: dict[str, float]` — verified literature vocalic nPVI (subset).
  - `class_separation(values: dict[str, float]) -> float` — mean(stress) - mean(syllable) over labelled languages; positive means correct ordering.
  - `spearman_against_reference(values: dict[str, float]) -> float` — Spearman r between computed values and `REFERENCE_NPVI_V` on shared languages.

- [ ] **Step 1: Write the failing test**

`tests/test_typology.py`:

```python
import pytest

from mol.validation import typology


def test_reference_covers_seed_languages():
    from mol.config import SEED_LANGUAGES
    assert set(typology.RHYTHM_CLASS) == set(SEED_LANGUAGES)


def test_class_separation_positive_when_stress_higher():
    # Stress-timed (english/german) high, syllable-timed (french/spanish) low.
    values = {"english": 60.0, "german": 58.0, "french": 40.0, "spanish": 30.0}
    sep = typology.class_separation(values)
    assert sep == pytest.approx(((60 + 58) / 2) - ((40 + 30) / 2))
    assert sep > 0


def test_spearman_against_reference_perfect_order():
    # Use the reference values themselves -> Spearman 1.0
    values = dict(typology.REFERENCE_NPVI_V)
    r = typology.spearman_against_reference(values)
    assert r == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_typology.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mol.validation.typology'`.

- [ ] **Step 3: Implement `src/mol/validation/__init__.py` (empty) and `src/mol/validation/typology.py`**

`src/mol/validation/__init__.py`:

```python
```

`src/mol/validation/typology.py`:

```python
"""Reference rhythm typology for the seed languages + agreement metrics.

Class labels follow the classical literature; Greek and Finnish are marked
'intermediate' (their classification is debated) and should be revisited in
exploration. Vocalic nPVI reference values are from Grabe & Low (2002) where
available. These are a validation *reference*, not ground truth.
"""

from __future__ import annotations

import math

from scipy.stats import spearmanr

RHYTHM_CLASS: dict[str, str] = {
    "english": "stress",
    "german": "stress",
    "polish": "stress",        # often classed stress-timed / mixed
    "french": "syllable",
    "spanish": "syllable",
    "italian": "syllable",
    "greek": "intermediate",
    "finnish": "intermediate",
}

# Verified vocalic nPVI values (Grabe & Low 2002). Subset only.
REFERENCE_NPVI_V: dict[str, float] = {
    "english": 57.2,
    "german": 59.7,
    "polish": 46.6,
    "french": 43.5,
    "spanish": 29.7,
}


def class_separation(values: dict[str, float]) -> float:
    """mean(stress-timed) - mean(syllable-timed) over labelled languages.

    Positive => stress-timed languages score higher (expected for vocalic nPVI).
    """
    stress = [v for k, v in values.items()
              if RHYTHM_CLASS.get(k) == "stress" and not math.isnan(v)]
    syllable = [v for k, v in values.items()
                if RHYTHM_CLASS.get(k) == "syllable" and not math.isnan(v)]
    if not stress or not syllable:
        return math.nan
    return sum(stress) / len(stress) - sum(syllable) / len(syllable)


def spearman_against_reference(values: dict[str, float]) -> float:
    """Spearman correlation between computed values and REFERENCE_NPVI_V."""
    shared = [k for k in REFERENCE_NPVI_V if k in values and not math.isnan(values[k])]
    if len(shared) < 3:
        return math.nan
    computed = [values[k] for k in shared]
    reference = [REFERENCE_NPVI_V[k] for k in shared]
    r, _ = spearmanr(computed, reference)
    return float(r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_typology.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite and commit**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all tests pass.

```bash
git add src/mol/validation/__init__.py src/mol/validation/typology.py tests/test_typology.py
git commit -m "feat: add rhythm-typology reference and agreement metrics"
```

---

### Task 14: Exploration notebook — fetch sample & extract features

**Files:**
- Create: `notebooks/01_explore_features.ipynb`
- Create: `scripts/collect_sample.py` (a thin, re-runnable CLI the notebook can also import)

**Deliverable (not TDD):** a small real-radio sample for the 8 seed languages under `data/`, a `data/lang_features.parquet` table, and an in-notebook 2-D rhythm-space plot. Acceptance: the table has one row per seed language with non-NaN `npvi_v_mean` and `percent_v_mean` for at least 6 of 8 languages.

- [ ] **Step 1: Write `scripts/collect_sample.py`**

```python
"""Collect a small real-radio sample and extract per-language prosody features.

Usage: python scripts/collect_sample.py --clips-per-lang 5 --clip-seconds 60
Writes data/clips/<lang>/*.wav and data/lang_features.parquet.
"""

from __future__ import annotations

import argparse

from mol.audio import load_audio, normalize_loudness
from mol.clean.vad import concat_speech, extract_speech, total_speech_seconds
from mol.config import DATA_DIR, SEED_LANGUAGES, TARGET_SAMPLE_RATE
from mol.features.aggregate import build_language_table
from mol.features.prosody_acoustic import ProsodyAcousticExtractor
from mol.ingest.radio import find_stations, record_clip


def collect(clips_per_lang: int, clip_seconds: int) -> None:
    extractor = ProsodyAcousticExtractor()
    per_language: dict[str, list[dict[str, float]]] = {}

    for key, spec in SEED_LANGUAGES.items():
        stations = find_stations(spec.radio_browser_lang, limit=clips_per_lang)
        vectors: list[dict[str, float]] = []
        for i, station in enumerate(stations[:clips_per_lang]):
            out = DATA_DIR / "clips" / key / f"{i:02d}.wav"
            try:
                record_clip(station.url, out, duration_s=clip_seconds)
                signal = normalize_loudness(load_audio(out))
                segments = extract_speech(signal)
                if total_speech_seconds(segments) < 5.0:
                    continue
                speech = concat_speech(signal, segments)
                vectors.append(extractor.extract(speech, sr=TARGET_SAMPLE_RATE))
            except Exception as exc:  # prototype: log and continue
                print(f"[skip] {key} station {i}: {exc}")
        if vectors:
            per_language[key] = vectors
        print(f"{key}: {len(vectors)} usable clips")

    table = build_language_table(per_language)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(DATA_DIR / "lang_features.parquet")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-per-lang", type=int, default=5)
    parser.add_argument("--clip-seconds", type=int, default=60)
    args = parser.parse_args()
    collect(args.clips_per_lang, args.clip_seconds)
```

- [ ] **Step 2: Run the collector (live network + ffmpeg)**

Run: `.venv/Scripts/python scripts/collect_sample.py --clips-per-lang 5 --clip-seconds 60`
Expected: prints per-language usable-clip counts and a feature table; writes `data/lang_features.parquet`.
If a language yields 0 clips, note it (station availability varies) and proceed; the gate doc records coverage.

- [ ] **Step 3: Create `notebooks/01_explore_features.ipynb`** with cells that:
  1. `import pandas as pd, matplotlib.pyplot as plt` and load `data/lang_features.parquet`.
  2. Scatter `percent_v_mean` (x) vs `npvi_v_mean` (y), annotating each point with the language name — the classic rhythm-space plot.
  3. Colour points by `mol.validation.typology.RHYTHM_CLASS`.
  4. Markdown cell: observations — do stress-timed languages sit higher on `npvi_v`? Any obvious data-quality issues (NaNs, outliers)?

- [ ] **Step 4: Commit**

```bash
git add scripts/collect_sample.py notebooks/01_explore_features.ipynb
git commit -m "feat: add sample collector and feature-exploration notebook"
```

(`data/` is gitignored; the parquet and audio are not committed.)

---

### Task 15: Validation notebook — typology agreement

**Files:**
- Create: `notebooks/02_validate_typology.ipynb`

**Deliverable (not TDD):** quantitative evidence on whether the alignment-free features reproduce known typology. Acceptance: notebook reports `class_separation` and `spearman_against_reference` for `npvi_v_mean`, plus a dendrogram, and a short written interpretation.

- [ ] **Step 1: Create the notebook** with cells that:
  1. Load `data/lang_features.parquet`.
  2. Build `values = df["npvi_v_mean"].to_dict()`; print `typology.class_separation(values)` and `typology.spearman_against_reference(values)`.
  3. Standardize the rhythm-relevant columns (`percent_v_mean, npvi_v_mean, rpvi_c_mean, varco_c_mean, varco_v_mean`), build a distance matrix, and draw a dendrogram (`scipy.cluster.hierarchy.dendrogram(distance.linkage_matrix(dm), labels=...)`).
  4. Draw the 2-D MDS scatter coloured by reference class.
  5. Markdown cell interpreting results: does clustering group by rhythm class? Where does it disagree, and is that signal or artefact (recall Arvaniti instability)?

- [ ] **Step 2: Run the notebook end-to-end** (Kernel → Restart & Run All). Confirm no errors and that the three figures render.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_validate_typology.ipynb
git commit -m "feat: add typology-validation notebook"
```

---

### Task 16: Heavy-method feasibility assessment

**Files:**
- Create: `notebooks/03_heavy_methods_feasibility.ipynb`

**Deliverable (not TDD):** a written, evidence-based assessment of the two heavier candidate methods we deliberately did NOT implement, so the gate decision considers all three fairly.

- [ ] **Step 1: Create the notebook** with markdown + small code probes covering:
  1. **Forced-alignment metrics (MFA):** what per-language models exist for the 8 seed languages, install/runtime cost on Windows, and whether transcripts are required. Probe: list MFA's available acoustic/dictionary models. Conclusion: infra weight vs. expected accuracy gain over the alignment-free %V/nPVI.
  2. **Learned envelope embedding (Deloche-style):** data/compute needed to train an envelope+voicing RNN; whether a pretrained checkpoint exists; how its embeddings would feed `proximity`. Probe: estimate clips/hours needed.
  3. A comparison table: method × (interpretability, infra cost, expected fidelity, blog-friendliness).

- [ ] **Step 2: Commit**

```bash
git add notebooks/03_heavy_methods_feasibility.ipynb
git commit -m "docs: add heavy-method feasibility assessment notebook"
```

---

### Task 17: Phase 0 findings & method-selection decision (THE GATE)

**Files:**
- Create: `docs/phase0-findings.md`

**Deliverable (not TDD):** the decision this whole phase exists to produce. This document is the hand-off into Phase 1 (alongside `docs/phase1-handoff.md`).

- [ ] **Step 1: Write `docs/phase0-findings.md`** covering, with concrete numbers/figures from the notebooks:
  1. **Data coverage:** clips and clean-speech seconds obtained per seed language; which languages were thin/missing and why.
  2. **Feature results:** the rhythm-space plot; per-language `npvi_v`, `percent_v`, varcos.
  3. **Typology agreement:** `class_separation` and `spearman_against_reference` values; dendrogram/MDS interpretation; where it matched and where it didn't.
  4. **Confound observations:** evidence of speech-rate / speaker / station effects (the Arvaniti risk) seen in the dispersion (`*_std`) columns.
  5. **Heavy-method assessment summary** (from Task 16).
  6. **DECISION:** which feature method(s) to carry into Phase 1 — alignment-free alone, alignment-free + a heavier method, or a combination — with the evidence that justifies it.
  7. **Recommended Phase 1 adjustments:** anything the data taught us (e.g. more clips/language, better station filtering, specific metrics to keep/drop).

- [ ] **Step 2: Update the spec's open questions** — in `docs/superpowers/specs/2026-06-24-music-of-languages-design.md` §11, append a line linking the resolved decision: `- Method selection resolved in docs/phase0-findings.md (Phase 0 gate).`

- [ ] **Step 3: Commit**

```bash
git add docs/phase0-findings.md docs/superpowers/specs/2026-06-24-music-of-languages-design.md
git commit -m "docs: record Phase 0 findings and method-selection decision"
```

---

## Self-Review

**Spec coverage** (against `2026-06-24-music-of-languages-design.md`):
- Modular `FeatureExtractor` interface → Task 5. ✅
- Alignment-free prosody/rhythm features (F0/intonation, tempo, %V/PVI/Varco) → Tasks 6–10. ✅
- Lightweight portable ingest/clean helpers (radio-browser + VAD + normalize), talk/news bias → Tasks 2–4, 14. ✅
- Per-language aggregation with dispersion (Arvaniti defense #1) → Task 11. ✅
- Validation against known typology (Arvaniti defense #2), kept as a reference not hard-wired → Task 13, 15. ✅
- Minimal proximity (distance/cluster/MDS) needed to evaluate methods → Task 12. ✅
- Seed languages (the agreed 8) → Task 1 config; used in Task 14. ✅
- Method-selection gate as the phase output → Task 17. ✅
- Consider all three candidate methods (impl. the light one; assess the heavy two) → Tasks 10 + 16. ✅
- Storage = plain local disk, no retention abstraction (deferred) → Global Constraints + `data/` gitignored. ✅
- NOT in Phase 0: diarization, ad removal, scaling, retention policy, colored maps → excluded by design; noted in handoff. ✅

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". The notebook/decision tasks (14–17) are inherently deliverable-based (exploration cannot be pre-written as asserted code), but each carries concrete acceptance criteria and exact cell contents. ✅

**Type consistency:** `FeatureVector = dict[str, float]` used consistently (Tasks 5, 10, 11). Feature key names defined in Tasks 8–10 (`npvi_v`, `percent_v`, `rpvi_c`, `varco_c`, …) match their `_mean`/`_std` aggregated forms used in Tasks 14–17 and in `typology` (Task 13). `distance.linkage_matrix`/`mds_2d`/`distance_matrix`/`standardize` names match between Task 12 and Tasks 15. `extract(signal, sr)` signature consistent across `base`, `prosody_acoustic`, and `collect_sample.py`. ✅

## Notes for the implementer

- **Run order:** Tasks 1–13 are pure/unit-testable and should be done first and in order (later tasks import earlier modules). Tasks 14–17 require live network + ffmpeg + the silero-vad model download and produce artifacts/decisions, not passing tests.
- **ffmpeg** must be on PATH for Task 14. On Windows: `winget install Gyan.FFmpeg` (or use the `! ` prefix in this session to run an interactive install).
- **First silero-vad / torch call** downloads a model; expect a one-time delay and network access.
- **If a language yields no usable clips** in Task 14, record it in the gate doc rather than blocking — station availability is exactly the kind of data-sourcing risk Phase 0 exists to surface.

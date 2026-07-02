# Workstream A — Source Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Survey and empirically probe candidate clip sources, then recommend a source mix + a Workstream B segment budget — delivered as `docs/source-evaluation.md`.

**Architecture:** A research phase (parallel subagents) produces a source-evaluation matrix + per-language capture *instances*. A small throwaway **probe harness** (`src/musiclang/probe/` + `scripts/probe_source.py`) then captures a handful of recordings per source per language and measures capturability + cleanliness (via the existing silero VAD) + independence. Synthesis assembles the matrix + measured probe numbers into the deliverable.

**Tech Stack:** Python 3.11+, uv, pytest (+ pytest-mock), pandas/pyarrow, silero-vad (existing `clean/vad.py`), ffmpeg, streamlink (HLS), feedparser (podcast RSS), HuggingFace `datasets` (VoxPopuli).

## Global Constraints

- **Run everything via `uv run`** (e.g. `uv run pytest`, `uv run python scripts/...`). uv uses managed CPython only (`python-preference = "only-managed"`).
- **Tests live flat in `tests/` as `test_<name>.py`**; pytest default deselects `slow` (`addopts = "-m 'not slow'"`). Any test that hits the real network or loads a real model MUST be marked `@pytest.mark.slow`. Default-run tests use **fakes / dependency injection** only (mirror `tests/test_radio.py`: `runner=`, `session=`, fake responses).
- **ffmpeg and streamlink are NOT on the tool-shell PATH.** Unit tests never invoke them (fakes). The single real-capture run (Task 6) must prepend the winget ffmpeg bin and use the venv's `streamlink`; this is handled there.
- **Audio working format: 16 kHz mono** — `TARGET_SAMPLE_RATE = 16_000` from `musiclang.config`.
- **Clean-speech bar: ≥ 30 s per recording** — `MIN_CLEAN_SPEECH_S = 30.0`.
- **8 seed languages, fixed:** keys of `musiclang.config.SEED_LANGUAGES` = english, german, polish, french, spanish, italian, greek, finnish.
- **Probe dependencies go in an optional `probe` extra** (`streamlink`, `feedparser`, `datasets`) — do not pollute core `dependencies`.
- **No LLM/ASR verification in this workstream** — that is Workstream C. Probes measure capturability + cleanliness + independence only.
- **Provenance (design spec §9):** cite each recommended source in the write-up. VoxPopuli = Wang et al. 2021, arXiv:2101.00390.
- **Branch:** `data-integrity-brief`. Commit after every task.
- **`data/` is gitignored** — all probe artifacts (`data/source_instances.parquet`, `data/source_probe_results.parquet`, `data/source_matrix_rows.json`, `data/probe_clips/`) stay local; never commit them.

## File Structure

| File | Responsibility |
|---|---|
| `src/musiclang/probe/__init__.py` | Package docstring; marks the throwaway probe harness. |
| `src/musiclang/probe/core.py` | Pure data + measurement: `RecordingRef`, `ProbeResult`, `measure_cleanliness`, `MIN_CLEAN_SPEECH_S`. No network. |
| `src/musiclang/probe/adapters.py` | Capture adapters (one ref → one 16k mono wav): progressive, HLS, RSS enclosure, corpus-local; `CAPTURE_DISPATCH`; `latest_enclosures`; `voxpopuli_probe`. |
| `scripts/probe_source.py` | Runner: expand instances → capture → measure → `summarize` → write results parquet. CLI. |
| `tests/test_probe_core.py` | Unit tests for `core.py`. |
| `tests/test_probe_adapters.py` | Unit tests for `adapters.py` (fakes for runner/downloader/parser/loader). |
| `tests/test_probe_runner.py` | Unit tests for `scripts/probe_source.py` helpers (`expand_instances`, `probe_ref`, `summarize`). |
| `docs/source-evaluation.md` | **Deliverable** — matrix + coverage sub-table + probe results + recommendation + B budget + verdict. |

**Finalists are fixed by capture mechanism, not chosen mid-plan.** The register stance (natural primary, corpus anchor) and the spec's expected mix make the three probed mechanisms certain: **podcast RSS**, **HLS/progressive radio**, and **VoxPopuli (corpus anchor)**. Broadcaster on-demand/news-API findings are folded into the radio (HLS) or podcast (RSS) instances where a broadcaster exposes them, and desk-noted otherwise. Task 7 documents the gates/ranking explicitly.

---

### Task 1: Desk survey — research subagents → matrix rows + capture instances

**Files:**
- Create: `data/source_instances.parquet` (gitignored) — per-language refs the probe will capture.
- Create: `data/source_matrix_rows.json` (gitignored) — matrix-row ratings + notes per source family.

**Interfaces:**
- Produces: `data/source_instances.parquet` with columns **`source`** (`radio`|`podcast`), **`language`** (seed key), **`channel_id`** (station/show id), **`kind`** (`progressive`|`hls`|`rss_feed`), **`ref`** (stream/m3u8/feed URL), **`notes`**. Consumed by `scripts/probe_source.py` (Task 5/6).
- Produces: `data/source_matrix_rows.json` — a list of `{source, coverage{lang->High|Med|Low|None}, ratings{criterion->High|Med|Low}, instances_count, register, legal, notes}`. Consumed by Task 7.

- [ ] **Step 1: Dispatch four research subagents in parallel (one message, four Agent calls)**

Use the Agent tool, `subagent_type: general-purpose` (web access). One per family. Give each this exact output contract:

> Return ONLY a JSON object: `{"source": "<radio|podcast|broadcaster_api|corpus>", "coverage": {"<lang>": "High|Med|Low|None", ... all 8}, "ratings": {"coverage":"H|M|L","cleanliness":"H|M|L","independence":"H|M|L","register":"H|M|L","capturability":"H|M|L","channel_diversity":"H|M|L","legal":"H|M|L"}, "register": "<spontaneous|read|mixed + 1 line>", "legal": "<license/ToS posture, 1-2 lines>", "instances": [{"language":"<key>","channel_id":"<id>","kind":"<progressive|hls|rss_feed>","ref":"<url>","notes":"<short>"}], "notes": "<≤120 words>"}`. Verify each URL resolves before listing it. Aim for ≥3 distinct channels/shows per language where the source serves it.

- Subagent **radio**: deepen radio-browser + non-radio-browser aggregators (Icecast/Shoutcast directory, per-country public-broadcaster station indexes); find **HLS `.m3u8`** talk/news streams for the Phase-0 gap languages (Finnish Yle, BBC English, German public, Italian RAI) and progressive talk streams elsewhere. `kind` = `hls` or `progressive`.
- Subagent **podcast**: find 3–6 talk/news/interview **podcast RSS feed URLs** per seed language (Podcastindex.org, Apple directory). `kind` = `rss_feed`, `ref` = feed URL.
- Subagent **broadcaster_api**: assess BBC Sounds, Yle Areena, RAI, ARD/ZDF, Radio France etc. — coverage, ToS, programmatic access. Where a broadcaster exposes a public **RSS feed** or **HLS** URL, emit it as an instance (so it gets probed under podcast/radio); else `instances: []` and desk-note.
- Subagent **corpus**: compare **VoxPopuli** vs **Common Voice** vs **MLS** on coverage of the 8, register, license, access. Recommend exactly one anchor. `instances: []` (corpus is probed via `voxpopuli_probe`, not URLs).

- [ ] **Step 2: Persist the matrix rows**

Write the four returned JSON objects as a list to `data/source_matrix_rows.json`.

- [ ] **Step 3: Persist the capture instances**

Flatten every subagent's `instances` into one table and write `data/source_instances.parquet`:

```python
import json, pandas as pd
from pathlib import Path
rows = json.loads(Path("data/source_matrix_rows.json").read_text(encoding="utf-8"))
inst = [
    {"source": ("podcast" if i["kind"] == "rss_feed" else "radio"),
     "language": i["language"], "channel_id": i["channel_id"],
     "kind": i["kind"], "ref": i["ref"], "notes": i.get("notes", "")}
    for fam in rows for i in fam.get("instances", [])
]
df = pd.DataFrame(inst, columns=["source", "language", "channel_id", "kind", "ref", "notes"])
df.to_parquet("data/source_instances.parquet")
print(df.groupby(["language", "kind"]).size())
```

- [ ] **Step 4: Verify coverage**

Run: `uv run python -c "import pandas as pd; df=pd.read_parquet('data/source_instances.parquet'); print(df['language'].value_counts()); print('langs:', sorted(df['language'].unique()))"`
Expected: every one of the 8 seed languages appears with ≥1 instance (gap languages may be radio-HLS-only or podcast-only — note any language that is thin; that is itself a finding, not a blocker).

- [ ] **Step 5: Commit** (only the gitignored-data note + any scratch; no data files are tracked — this commit is a no-op marker, so skip if `git status` is clean and proceed.)

```bash
git status   # data/ is gitignored; nothing to commit here
```

---

### Task 2: Probe core — `RecordingRef`, `ProbeResult`, `measure_cleanliness`

**Files:**
- Create: `src/musiclang/probe/__init__.py`
- Create: `src/musiclang/probe/core.py`
- Test: `tests/test_probe_core.py`

**Interfaces:**
- Produces: `RecordingRef(source, language, channel_id, kind, ref)` (frozen dataclass); `ProbeResult(source, language, channel_id, kind, capturable, clean_speech_s, meets_30s, error="")`; `measure_cleanliness(signal, sr=TARGET_SAMPLE_RATE, *, speech_fn=extract_speech) -> tuple[float, bool]`; `MIN_CLEAN_SPEECH_S = 30.0`. Consumed by `adapters.py` and `scripts/probe_source.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_core.py
import numpy as np
from musiclang.probe.core import (
    RecordingRef, ProbeResult, measure_cleanliness, MIN_CLEAN_SPEECH_S,
)


def test_measure_cleanliness_sums_speech_and_flags_30s():
    signal = np.zeros(16_000 * 40, dtype=np.float32)
    # injected VAD: 20 s + 15 s = 35 s of speech, clears the 30 s bar
    clean_s, meets = measure_cleanliness(
        signal, speech_fn=lambda sig, sr: [(0.0, 20.0), (25.0, 40.0)]
    )
    assert clean_s == 35.0
    assert meets is True


def test_measure_cleanliness_below_threshold_not_flagged():
    signal = np.zeros(10, dtype=np.float32)
    clean_s, meets = measure_cleanliness(signal, speech_fn=lambda sig, sr: [(0.0, 12.0)])
    assert clean_s == 12.0
    assert meets is False
    assert MIN_CLEAN_SPEECH_S == 30.0


def test_recordingref_and_proberesult_fields():
    ref = RecordingRef("radio", "finnish", "yle-1", "hls", "http://x.m3u8")
    assert (ref.source, ref.kind, ref.ref) == ("radio", "hls", "http://x.m3u8")
    res = ProbeResult("radio", "finnish", "yle-1", "hls", True, 33.0, True)
    assert res.capturable and res.meets_30s and res.error == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'musiclang.probe'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/probe/__init__.py
"""Throwaway source-probe harness (Data Integrity phase, Workstream A).

Measures capturability + cleanliness + independence of candidate clip sources.
Deliberately minimal — NOT the Phase-1 ingest package.
"""
```

```python
# src/musiclang/probe/core.py
"""Pure data + measurement for the source probe (no network, no models)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from musiclang.clean.vad import extract_speech, total_speech_seconds
from musiclang.config import TARGET_SAMPLE_RATE

MIN_CLEAN_SPEECH_S: float = 30.0


@dataclass(frozen=True)
class RecordingRef:
    source: str       # 'radio' | 'podcast' | 'corpus'
    language: str     # seed-language key, e.g. 'finnish'
    channel_id: str   # station / show / speaker id — for distinctness counting
    kind: str         # 'progressive' | 'hls' | 'rss' | 'corpus'
    ref: str          # stream / m3u8 / enclosure url, or local wav path (corpus)


@dataclass(frozen=True)
class ProbeResult:
    source: str
    language: str
    channel_id: str
    kind: str
    capturable: bool
    clean_speech_s: float
    meets_30s: bool
    error: str = ""


def measure_cleanliness(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, *, speech_fn=extract_speech
) -> tuple[float, bool]:
    """Seconds of clean speech in `signal` and whether it clears the 30 s bar.

    `speech_fn(signal, sr) -> list[(start_s, end_s)]` is injectable so unit tests
    avoid loading the silero model; production passes the real `extract_speech`.
    """
    segments = speech_fn(signal, sr)
    clean_s = total_speech_seconds(segments)
    return clean_s, clean_s >= MIN_CLEAN_SPEECH_S
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_core.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/probe/__init__.py src/musiclang/probe/core.py tests/test_probe_core.py
git commit -m "feat(probe): RecordingRef/ProbeResult + measure_cleanliness core"
```

---

### Task 3: Capture adapters — progressive / HLS / RSS + dispatch + enclosures

**Files:**
- Create: `src/musiclang/probe/adapters.py`
- Modify: `pyproject.toml` (add optional `probe` extra: `streamlink`, `feedparser`)
- Test: `tests/test_probe_adapters.py`

**Interfaces:**
- Consumes: `RecordingRef` from `probe.core`; `record_clip` from `ingest.radio`.
- Produces: `capture_progressive(ref, out_path, *, duration_s=60, runner=subprocess.run) -> Path|None`; `capture_hls(ref, out_path, *, duration_s=60, runner=subprocess.run) -> Path|None`; `capture_rss(ref, out_path, *, skip_s=30, take_s=90, downloader=_download, runner=subprocess.run) -> Path|None`; `capture_local(ref, out_path, **_) -> Path|None`; `CAPTURE_DISPATCH: dict[str, callable]`; `latest_enclosures(feed_url, k, *, parser=feedparser.parse) -> list[str]`. Consumed by `scripts/probe_source.py`.

- [ ] **Step 1: Add the probe dependency extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add a `probe` group alongside the existing `dev`/`feasibility`:

```toml
probe = ["streamlink>=6.0", "feedparser>=6.0"]
```

Then: `uv sync --extra probe --extra dev`
Expected: resolves and installs streamlink + feedparser into the venv.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_probe_adapters.py
from pathlib import Path
from types import SimpleNamespace

from musiclang.probe import adapters
from musiclang.probe.core import RecordingRef


def _fake_runner(captured):
    def run(cmd, **kwargs):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFFfake")
        captured.append(cmd)
        return SimpleNamespace(returncode=0)
    return run


def test_capture_hls_builds_streamlink_then_ffmpeg(tmp_path):
    cmds = []
    ref = RecordingRef("radio", "finnish", "yle", "hls", "http://x.m3u8")
    out = tmp_path / "o.wav"
    res = adapters.capture_hls(ref, out, duration_s=60, runner=_fake_runner(cmds))
    assert res == out and out.exists()
    assert cmds[0][0] == "streamlink" and "http://x.m3u8" in cmds[0] and "00:01:00" in cmds[0]
    assert cmds[1][0] == "ffmpeg" and cmds[1][-1] == str(out)


def test_capture_rss_downloads_then_slices(tmp_path):
    cmds = []
    def fake_dl(url, dest):
        Path(dest).write_bytes(b"media")
        return dest
    ref = RecordingRef("podcast", "french", "show", "rss", "http://ep.mp3")
    out = tmp_path / "o.wav"
    res = adapters.capture_rss(ref, out, downloader=fake_dl, runner=_fake_runner(cmds))
    assert res == out
    ff = cmds[0]
    assert ff[0] == "ffmpeg" and "-ss" in ff and "-t" in ff


def test_capture_rss_returns_none_when_download_fails(tmp_path):
    ref = RecordingRef("podcast", "french", "show", "rss", "http://ep.mp3")
    res = adapters.capture_rss(
        ref, tmp_path / "o.wav", downloader=lambda u, d: None, runner=lambda *a, **k: None
    )
    assert res is None


def test_capture_local_returns_existing_path(tmp_path):
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"x")
    ref = RecordingRef("corpus", "greek", "spk1", "corpus", str(wav))
    assert adapters.capture_local(ref, tmp_path / "ignored.wav") == wav


def test_latest_enclosures_extracts_audio_hrefs():
    feed = SimpleNamespace(entries=[
        SimpleNamespace(enclosures=[{"href": "http://a.mp3", "type": "audio/mpeg"}]),
        SimpleNamespace(enclosures=[{"href": "http://b.jpg", "type": "image/jpeg"}]),
        SimpleNamespace(enclosures=[{"href": "http://c.mp3", "type": "audio/mpeg"}]),
    ])
    urls = adapters.latest_enclosures("http://feed", 2, parser=lambda u: feed)
    assert urls == ["http://a.mp3", "http://c.mp3"]


def test_capture_dispatch_maps_kinds():
    assert set(adapters.CAPTURE_DISPATCH) == {"progressive", "hls", "rss", "corpus"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_adapters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'musiclang.probe.adapters'`.

- [ ] **Step 4: Write minimal implementation**

```python
# src/musiclang/probe/adapters.py
"""Capture adapters: turn one RecordingRef into one 16 kHz mono wav.

Throwaway-quality probes. ffmpeg/streamlink are invoked via an injectable
`runner` so unit tests never shell out (mirrors ingest.radio.record_clip).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import requests

from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.ingest.radio import record_clip
from musiclang.probe.core import RecordingRef

USER_AGENT = "music-of-languages/0.0 (research; contact: project owner)"


def _seconds_to_hhmmss(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def capture_progressive(
    ref: RecordingRef, out_path, *, duration_s: int = 60, runner=subprocess.run
) -> Path | None:
    """Capture a progressive (icecast/shoutcast) stream via ffmpeg (reuses record_clip)."""
    try:
        return record_clip(ref.ref, out_path, duration_s=duration_s, runner=runner)
    except Exception:
        return None


def capture_hls(
    ref: RecordingRef, out_path, *, duration_s: int = 60, runner=subprocess.run
) -> Path | None:
    """Capture an HLS (.m3u8) slice with streamlink, then transcode to 16k mono wav."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        media = Path(tmp) / "hls.ts"
        streamlink_cmd = [
            "streamlink", "--hls-live-restart",
            "--hls-duration", _seconds_to_hhmmss(duration_s),
            "-o", str(media), ref.ref, "best",
        ]
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(media),
            "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), str(out_path),
        ]
        try:
            runner(streamlink_cmd, check=True)
            if not media.exists():
                return None
            runner(ffmpeg_cmd, check=True)
        except Exception:
            return None
    return out_path if out_path.exists() else None


def _download(url: str, dest: Path) -> Path | None:
    try:
        with requests.get(
            url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}
        ) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        return dest
    except Exception:
        return None


def capture_rss(
    ref: RecordingRef, out_path, *, skip_s: int = 30, take_s: int = 90,
    downloader=None, runner=subprocess.run,
) -> Path | None:
    """Download an episode enclosure, transcode the [skip_s, skip_s+take_s] slice to wav.

    The skip skips a likely intro jingle so the measured slice is mostly talk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    downloader = downloader or _download
    with tempfile.TemporaryDirectory() as tmp:
        media = Path(tmp) / "episode.media"
        if downloader(ref.ref, media) is None:
            return None
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(skip_s), "-t", str(take_s), "-i", str(media),
            "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), str(out_path),
        ]
        try:
            runner(ffmpeg_cmd, check=True)
        except Exception:
            return None
    return out_path if out_path.exists() else None


def capture_local(ref: RecordingRef, out_path=None, **_) -> Path | None:
    """Corpus refs already point at a written wav — just hand it back if present."""
    p = Path(ref.ref)
    return p if p.exists() else None


def latest_enclosures(feed_url: str, k: int, *, parser=None) -> list[str]:
    """Up to k audio enclosure URLs from a podcast RSS feed (feed order, newest first)."""
    if parser is None:
        import feedparser
        parser = feedparser.parse
    feed = parser(feed_url)
    urls: list[str] = []
    for entry in getattr(feed, "entries", [])[: k * 3]:
        for enc in getattr(entry, "enclosures", []) or []:
            href = enc.get("href") or enc.get("url")
            if href and "audio" in (enc.get("type", "") or ""):
                urls.append(href)
                break
        if len(urls) >= k:
            break
    return urls[:k]


CAPTURE_DISPATCH = {
    "progressive": capture_progressive,
    "hls": capture_hls,
    "rss": capture_rss,
    "corpus": capture_local,
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_adapters.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/musiclang/probe/adapters.py tests/test_probe_adapters.py pyproject.toml uv.lock
git commit -m "feat(probe): capture adapters (progressive/hls/rss/local) + enclosures"
```

---

### Task 4: Corpus probe — `voxpopuli_probe`

**Files:**
- Modify: `src/musiclang/probe/adapters.py` (add `voxpopuli_probe`, `_VOXPOPULI_LANG`, `_load_voxpopuli`)
- Modify: `pyproject.toml` (`probe` extra: add `datasets`)
- Test: `tests/test_probe_adapters.py` (add corpus tests)

**Interfaces:**
- Produces: `voxpopuli_probe(language, n, out_dir, *, loader=_load_voxpopuli, writer=soundfile.write) -> list[tuple[RecordingRef, Path]]` — pulls up to `n` **distinct-speaker** VoxPopuli samples, writes 16k mono wavs, returns `(RecordingRef(kind='corpus', ref=<wav path>), path)` pairs. Consumed by `scripts/probe_source.py`.

- [ ] **Step 1: Add the datasets dependency**

In `pyproject.toml`, extend the `probe` extra:

```toml
probe = ["streamlink>=6.0", "feedparser>=6.0", "datasets>=2.14"]
```

Then: `uv sync --extra probe --extra dev`

- [ ] **Step 2: Write the failing test** (append to `tests/test_probe_adapters.py`)

```python
def test_voxpopuli_probe_dedups_speakers_and_writes(tmp_path):
    items = [
        {"speaker_id": "s1", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
        {"speaker_id": "s1", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},  # dup
        {"speaker_id": "s2", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
        {"speaker_id": "s3", "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
    ]
    written = []
    def fake_writer(path, arr, sr):
        Path(path).write_bytes(b"wav")
        written.append((path, sr))
    pairs = adapters.voxpopuli_probe(
        "finnish", 2, tmp_path, loader=lambda lang: iter(items), writer=fake_writer
    )
    assert len(pairs) == 2
    assert [ref.channel_id for ref, _ in pairs] == ["s1", "s2"]   # deduped, capped at n
    assert all(ref.kind == "corpus" and ref.source == "corpus" for ref, _ in pairs)
    assert all(Path(p).exists() for _, p in pairs)
    assert written[0][1] == 16_000


def test_voxpopuli_probe_maps_all_eight_languages():
    from musiclang.config import SEED_LANGUAGES
    assert set(adapters._VOXPOPULI_LANG) == set(SEED_LANGUAGES)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_adapters.py -k voxpopuli -v`
Expected: FAIL with `AttributeError: module 'musiclang.probe.adapters' has no attribute 'voxpopuli_probe'`.

- [ ] **Step 4: Write minimal implementation** (append to `src/musiclang/probe/adapters.py`)

```python
_VOXPOPULI_LANG = {
    "english": "en", "german": "de", "polish": "pl", "french": "fr",
    "spanish": "es", "italian": "it", "greek": "el", "finnish": "fi",
}


def _load_voxpopuli(language: str):
    """Streaming VoxPopuli test split for `language` (Wang et al. 2021, arXiv:2101.00390)."""
    from datasets import load_dataset
    code = _VOXPOPULI_LANG[language]
    return load_dataset("facebook/voxpopuli", code, split="test", streaming=True)


def voxpopuli_probe(
    language: str, n: int, out_dir, *, loader=None, writer=None
) -> list[tuple[RecordingRef, Path]]:
    """Pull up to `n` distinct-speaker VoxPopuli samples; write 16k mono wavs.

    Distinct `speaker_id` per sample approximates one-independent-recording-each.
    """
    import soundfile as sf

    loader = loader or _load_voxpopuli
    writer = writer or sf.write
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[RecordingRef, Path]] = []
    seen: set[str] = set()
    for item in loader(language):
        spk = str(item.get("speaker_id", "") or f"spk{len(pairs)}")
        if spk in seen:
            continue
        seen.add(spk)
        audio = item["audio"]
        wav = out_dir / f"voxpopuli_{language}_{len(pairs):02d}.wav"
        writer(str(wav), audio["array"], audio["sampling_rate"])
        ref = RecordingRef("corpus", language, spk, "corpus", str(wav))
        pairs.append((ref, wav))
        if len(pairs) >= n:
            break
    return pairs
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_adapters.py -k voxpopuli -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/musiclang/probe/adapters.py tests/test_probe_adapters.py pyproject.toml uv.lock
git commit -m "feat(probe): voxpopuli_probe corpus adapter (distinct-speaker pull)"
```

---

### Task 5: Probe runner — `scripts/probe_source.py`

**Files:**
- Create: `scripts/probe_source.py`
- Test: `tests/test_probe_runner.py`

**Interfaces:**
- Consumes: `RecordingRef`, `ProbeResult`, `measure_cleanliness` (core); `CAPTURE_DISPATCH`, `latest_enclosures`, `voxpopuli_probe` (adapters); `load_audio`, `normalize_loudness` (`musiclang.audio`).
- Produces (module-level, importable for tests): `expand_instances(rows, k) -> list[RecordingRef]`; `probe_ref(ref, work_dir, *, capture_dispatch=CAPTURE_DISPATCH, load=load_audio, normalize=normalize_loudness, speech_fn=extract_speech) -> ProbeResult`; `summarize(results) -> pd.DataFrame`; `run(n_per_language, sources, instances_path, out_path) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_runner.py
import importlib.util
from pathlib import Path

import numpy as np

# scripts/ is not a package — load the module by path.
_SPEC = importlib.util.spec_from_file_location(
    "probe_source", Path(__file__).resolve().parents[1] / "scripts" / "probe_source.py"
)
probe_source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe_source)

from musiclang.probe.core import RecordingRef, ProbeResult  # noqa: E402


def test_expand_instances_expands_rss_feeds_and_passes_others():
    rows = [
        {"source": "podcast", "language": "french", "channel_id": "showA",
         "kind": "rss_feed", "ref": "http://feedA"},
        {"source": "radio", "language": "finnish", "channel_id": "yle",
         "kind": "hls", "ref": "http://x.m3u8"},
    ]
    refs = probe_source.expand_instances(
        rows, k=2, enclosure_fn=lambda url, k: ["http://e1.mp3", "http://e2.mp3"]
    )
    kinds = [r.kind for r in refs]
    assert kinds.count("rss") == 2 and "hls" in kinds
    rss = [r for r in refs if r.kind == "rss"]
    assert rss[0].source == "podcast" and rss[0].ref == "http://e1.mp3"
    assert rss[0].channel_id == "showA"


def test_probe_ref_measures_when_capture_succeeds(tmp_path):
    ref = RecordingRef("radio", "greek", "st1", "progressive", "http://s")
    def fake_capture(r, out, **k):
        Path(out).write_bytes(b"x")
        return Path(out)
    res = probe_source.probe_ref(
        ref, tmp_path,
        capture_dispatch={"progressive": fake_capture},
        load=lambda p: np.zeros(16_000 * 40, dtype=np.float32),
        normalize=lambda s: s,
        speech_fn=lambda sig, sr: [(0.0, 33.0)],
    )
    assert res.capturable is True and res.meets_30s is True and res.clean_speech_s == 33.0


def test_probe_ref_reports_capture_failure(tmp_path):
    ref = RecordingRef("radio", "greek", "st1", "hls", "http://s")
    res = probe_source.probe_ref(
        ref, tmp_path, capture_dispatch={"hls": lambda r, o, **k: None}
    )
    assert res.capturable is False and res.error == "capture-failed"


def test_summarize_aggregates_per_source_language():
    results = [
        ProbeResult("radio", "greek", "a", "hls", True, 33.0, True),
        ProbeResult("radio", "greek", "b", "hls", True, 10.0, False),
        ProbeResult("radio", "greek", "a", "hls", False, 0.0, False, "capture-failed"),
    ]
    df = probe_source.summarize(results)
    row = df[(df["source"] == "radio") & (df["language"] == "greek")].iloc[0]
    assert row["n"] == 3
    assert abs(row["capture_rate"] - 2 / 3) < 1e-9
    assert row["distinct_channels"] == 2          # 'a' and 'b'
    assert row["median_clean_s"] == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_runner.py -v`
Expected: FAIL — `scripts/probe_source.py` does not exist (spec load raises `FileNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/probe_source.py
"""Probe candidate clip sources: capture a few recordings/source/language and
measure capturability + cleanliness + independence. Workstream A (throwaway).

Usage:
    uv run python scripts/probe_source.py [--n-per-language 4]
        [--sources radio,podcast,corpus] [--instances data/source_instances.parquet]
        [--out data/source_probe_results.parquet]

Requires ffmpeg + streamlink reachable on PATH for real runs (see the plan's
Task 6 for the Windows PATH handling). `data/` is gitignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from musiclang.audio import load_audio, normalize_loudness
from musiclang.clean.vad import extract_speech
from musiclang.config import DATA_DIR, SEED_LANGUAGES, TARGET_SAMPLE_RATE
from musiclang.probe import adapters
from musiclang.probe.core import ProbeResult, RecordingRef, measure_cleanliness


def expand_instances(rows, k: int, *, enclosure_fn=None) -> list[RecordingRef]:
    """Instance rows -> RecordingRefs; expand each 'rss_feed' into up to k episodes."""
    enclosure_fn = enclosure_fn or adapters.latest_enclosures
    refs: list[RecordingRef] = []
    for row in rows:
        if row["kind"] == "rss_feed":
            for url in enclosure_fn(row["ref"], k):
                refs.append(RecordingRef(
                    row["source"], row["language"], row["channel_id"], "rss", url
                ))
        else:
            refs.append(RecordingRef(
                row["source"], row["language"], row["channel_id"], row["kind"], row["ref"]
            ))
    return refs


def probe_ref(
    ref: RecordingRef, work_dir, *, capture_dispatch=None,
    load=load_audio, normalize=normalize_loudness, speech_fn=extract_speech,
) -> ProbeResult:
    """Capture one ref, measure clean-speech seconds. Never raises — errors -> ProbeResult."""
    capture_dispatch = capture_dispatch if capture_dispatch is not None else adapters.CAPTURE_DISPATCH
    fn = capture_dispatch.get(ref.kind)
    safe_id = ref.channel_id.replace("/", "_")[:40]
    out = Path(work_dir) / f"{ref.source}_{ref.language}_{safe_id}_{ref.kind}.wav"
    try:
        path = fn(ref, out) if fn is not None else None
        if path is None:
            return ProbeResult(ref.source, ref.language, ref.channel_id, ref.kind,
                               False, 0.0, False, "capture-failed")
        signal = normalize(load(path))
        clean_s, meets = measure_cleanliness(signal, TARGET_SAMPLE_RATE, speech_fn=speech_fn)
        return ProbeResult(ref.source, ref.language, ref.channel_id, ref.kind,
                           True, float(clean_s), bool(meets))
    except Exception as exc:  # noqa: BLE001 — a probe failure is data, not a crash
        return ProbeResult(ref.source, ref.language, ref.channel_id, ref.kind,
                           False, 0.0, False, str(exc)[:200])


def summarize(results: list[ProbeResult]) -> pd.DataFrame:
    """Per (source, language): N, capture rate, 30s-meet rate, median clean s, distinct channels."""
    if not results:
        return pd.DataFrame(
            columns=["source", "language", "n", "capture_rate",
                     "meets_30s_rate", "median_clean_s", "distinct_channels"]
        )
    df = pd.DataFrame([r.__dict__ for r in results])
    grouped = df.groupby(["source", "language"])
    out = grouped.agg(
        n=("capturable", "size"),
        capture_rate=("capturable", "mean"),
        meets_30s_rate=("meets_30s", "mean"),
        median_clean_s=("clean_speech_s", "median"),
        distinct_channels=("channel_id", "nunique"),
    ).reset_index()
    return out


def run(
    n_per_language: int = 4,
    sources: tuple[str, ...] = ("radio", "podcast", "corpus"),
    instances_path: str | Path = None,
    out_path: str | Path = None,
) -> pd.DataFrame:
    instances_path = Path(instances_path or DATA_DIR / "source_instances.parquet")
    out_path = Path(out_path or DATA_DIR / "source_probe_results.parquet")
    work_dir = DATA_DIR / "probe_clips"
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[ProbeResult] = []

    # --- radio + podcast: from the research instances file ---
    if instances_path.exists():
        inst = pd.read_parquet(instances_path)
        inst = inst[inst["source"].isin([s for s in sources if s in ("radio", "podcast")])]
        refs = expand_instances(inst.to_dict("records"), k=n_per_language)
        for ref in refs:
            res = probe_ref(ref, work_dir)
            results.append(res)
            print(f"[{res.source}/{res.language}] {res.channel_id}: "
                  f"{'ok' if res.capturable else res.error} clean={res.clean_speech_s:.1f}s")

    # --- corpus: VoxPopuli, distinct speakers per language ---
    if "corpus" in sources:
        for lang in SEED_LANGUAGES:
            try:
                pairs = adapters.voxpopuli_probe(lang, n_per_language, work_dir / "corpus" / lang)
            except Exception as exc:  # noqa: BLE001
                print(f"[corpus/{lang}] load failed: {exc}")
                continue
            for ref, _wav in pairs:
                res = probe_ref(ref, work_dir)
                results.append(res)
                print(f"[corpus/{lang}] {res.channel_id}: clean={res.clean_speech_s:.1f}s")

    df = pd.DataFrame([r.__dict__ for r in results])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    summary = summarize(results)
    print("\n=== per-source/language summary ===")
    print(summary.to_string(index=False))
    summary.to_parquet(out_path.with_name("source_probe_summary.parquet"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe candidate clip sources.")
    parser.add_argument("--n-per-language", type=int, default=4)
    parser.add_argument("--sources", type=str, default="radio,podcast,corpus")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    run(
        n_per_language=args.n_per_language,
        sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
        instances_path=args.instances,
        out_path=args.out,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_runner.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the whole suite (no regressions)**

Run: `uv run pytest -q`
Expected: all pass (probe tests added, existing tests untouched).

- [ ] **Step 6: Commit**

```bash
git add scripts/probe_source.py tests/test_probe_runner.py
git commit -m "feat(probe): source-probe runner (expand/probe/summarize) + CLI"
```

---

### Task 6: Run the probes (controller-driven, all 8 languages)

**Files:**
- Produces: `data/source_probe_results.parquet`, `data/source_probe_summary.parquet` (gitignored).

This task runs real captures. **It is controller-driven as a background job** (per the project memory note: drive long-running capture/inference as a controller-owned background job, not inside a subagent). No new code.

- [ ] **Step 1: Confirm instances exist from Task 1**

Run: `uv run python -c "import pandas as pd; print(pd.read_parquet('data/source_instances.parquet').groupby(['language','kind']).size())"`
Expected: instances for the 8 languages (radio/podcast). If missing, return to Task 1.

- [ ] **Step 2: Make ffmpeg + streamlink reachable, then launch in background**

ffmpeg is installed via winget but not on the tool-shell PATH; streamlink is in the venv (Task 3). Locate ffmpeg, prepend it, and launch (PowerShell):

```powershell
$ff = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ff) { $env:PATH = (Split-Path $ff.FullName) + ";" + $env:PATH }
ffmpeg -version | Select-Object -First 1     # verify resolves
uv run python scripts/probe_source.py --n-per-language 4 --sources radio,podcast,corpus
```

Run this with the Bash/PowerShell tool's **`run_in_background: true`**. Monitor to completion (it makes real network fetches across 8 languages × 3 sources; VoxPopuli streaming + radio/podcast captures take time). If ffmpeg is not found under WinGet, find it with `(Get-Command ffmpeg).Source` in the user's interactive shell and prepend that directory instead.

- [ ] **Step 3: Inspect the results**

Run: `uv run python -c "import pandas as pd; print(pd.read_parquet('data/source_probe_summary.parquet').to_string(index=False))"`
Expected: a per-(source,language) table with `capture_rate`, `meets_30s_rate`, `median_clean_s`, `distinct_channels`. Sanity: corpus (VoxPopuli) should show high capture_rate + high meets_30s; radio/podcast vary by language (gap languages are the interesting rows). Record any source/language that captured nothing — that is a finding for Task 7.

- [ ] **Step 4: No commit** (results are gitignored data). Proceed to Task 7.

---

### Task 7: Synthesize + write `docs/source-evaluation.md`

**Files:**
- Create: `docs/source-evaluation.md` (the deliverable — committed).
- Read: `data/source_matrix_rows.json` (Task 1), `data/source_probe_summary.parquet` (Task 6).

**Interfaces:** none (terminal deliverable).

- [ ] **Step 1: Load both inputs**

```python
import json, pandas as pd
from pathlib import Path
rows = json.loads(Path("data/source_matrix_rows.json").read_text(encoding="utf-8"))
summary = pd.read_parquet("data/source_probe_summary.parquet")
print(summary.to_string(index=False))
for fam in rows:
    print(fam["source"], fam["ratings"], fam["coverage"])
```

- [ ] **Step 2: Write `docs/source-evaluation.md`** with these sections (filled from the two inputs — no placeholders):

1. **Summary** — the recommended mix in two sentences + the headline B segment budget.
2. **Source-evaluation matrix** — a table: rows = the four families (radio, podcast, broadcaster_api, corpus), columns = the 7 criteria (coverage, cleanliness, independence, register, capturability, channel diversity, legal), cells = High/Med/Low from `ratings` + a short note.
3. **Per-language coverage sub-table** — rows = the 8 languages, columns = the four families, cells = High/Med/Low/None from each family's `coverage`.
4. **Finalist probe results** — the `summarize` table (source × language: capture_rate, meets_30s_rate, median_clean_s, distinct_channels) for the three probed mechanisms.
5. **Decision rule applied** — state the four hard gates (covers/extensible-to the 8 · capturable to ≥30 s clean · many independent recordings · legal acceptable), which families pass, and the High/Med/Low ranking of survivors; name the finalists (podcast / HLS-progressive radio / VoxPopuli) and why broadcaster-API content folds into radio/podcast.
6. **Recommendation** — the mix, **per-language** where coverage demands it (e.g. podcasts for breadth + HLS/broadcaster radio for gap languages + VoxPopuli anchor), tied to the gates/ranking + probe numbers.
7. **Workstream B segment budget** — target independent segments/language and minimum distinct channels/language, justified by the probe `capture_rate`/`distinct_channels` (e.g. "Finnish radio captured X% over Y channels → cap Finnish at N segments, lean on podcasts + VoxPopuli").
8. **Build-now-vs-defer verdict** — which fetchers Workstream B must build (HLS? podcast RSS? VoxPopuli client?) vs defer to Phase 1, based on which sources the recommendation actually uses.
9. **Legal/attribution notes** (from each family's `legal`; cite VoxPopuli = Wang et al. 2021, arXiv:2101.00390) + **honest caveats** (the read-register anchor trade-off; any language whose coverage stays thin after probing).

- [ ] **Step 3: Self-check the deliverable**

Re-read `docs/source-evaluation.md`: every section filled from real data (no "TBD"); the matrix has all 4 families × 7 criteria; the coverage sub-table has all 8 languages; the probe table has measured numbers; the recommendation names a concrete per-language mix + a numeric B budget + a build/defer verdict.

- [ ] **Step 4: Commit**

```bash
git add docs/source-evaluation.md
git commit -m "docs: Workstream A source-evaluation matrix + recommended mix + B budget"
```

---

## Self-Review

**Spec coverage** (each spec section → task):
- §2 decision 1 (hybrid: survey + finalist probes) → Tasks 1 (survey) + 2–6 (probe). ✓
- §2 decision 2 (natural primary, corpus anchor) → Task 1 corpus subagent recommends one anchor; Task 7 §6/§9 frames natural-primary + flags register. ✓
- §2 decision 3 (four families, YouTube excluded) → Task 1 four subagents; no YouTube path anywhere. ✓
- §2 decision 4 (all 8 languages in probe) → Task 6 runs all 8; Task 5 `run()` loops `SEED_LANGUAGES`. ✓
- §4 candidate set → Task 1 subagent briefs (radio incl. aggregators/HLS, podcasts, broadcaster APIs, VoxPopuli vs CV vs MLS). ✓
- §5 criteria (✎ desk / ⚙ probe) → desk = Task 1 `ratings`; probe ⚙ (cleanliness, capturability, independence) = Tasks 2–6. ✓
- §6 two-tier method → Task 1 (tier 1) + Tasks 2–6 (tier 2). ✓
- §7 decision rule (gates + ranking → per-language mix + B budget + build/defer) → Task 7 §5–§8. ✓
- §8 components (adapters + runner, behind stage boundaries, minimal) → Tasks 2–5 (reuse `record_clip`, `vad.py`, `audio.py`). ✓
- §9 execution (parallel research → synthesis → controller-driven probes → write-up) → Tasks 1 → 6→7, probes background in Task 6. ✓
- §10 deliverable structure → Task 7 Step 2 mirrors the spec's 8-part structure (+ decision-rule section). ✓
- §11 risks (probe flakiness as data; corpus register flagged; no scope creep into B; legal noted) → `probe_ref` never raises (Task 5); Task 7 §9 caveats; probes capped at `--n-per-language`, no LLM. ✓

**Placeholder scan:** no "TBD"/"add error handling"/"similar to Task N" — every code + test block is complete. ✓

**Type consistency:** `RecordingRef(source, language, channel_id, kind, ref)` and `ProbeResult(...)` identical across Tasks 2/3/4/5; `measure_cleanliness(signal, sr, *, speech_fn)` signature matches its call in `probe_ref`; `CAPTURE_DISPATCH` keys `{progressive,hls,rss,corpus}` match `kind` values produced by `expand_instances`/`voxpopuli_probe`; `summarize` columns match the Task 7 probe-table description. ✓

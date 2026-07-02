# Workstream B+C — Verified Independent-Segment Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect one independent ~30 s clean-speech segment per recording (podcast-primary + radio) and audio-verify each (hybrid AST tagger → Whisper language-ID → LLM tie-break), producing a verified-segment dataset + provenance manifest + per-segment verdicts.

**Architecture:** A pure `select_segment` (first 30 s of concatenated clean speech) + a `verify/` module (a hybrid short-circuit `verify_segment` over injectable tagger/whisper/llm callables) + an extended provenance manifest, wired by a breadth collector (`scripts/collect_segments.py`) that reuses the proven `probe/adapters.py` capture functions and `clean_clip`. A **pilot** (1 verified segment/language) is manually reviewed by the user before the full ≥25/language run.

**Tech Stack:** Python 3.11+, uv, pytest (+pytest-mock), numpy/pandas, silero-vad (existing `clean/vad`), `transformers` AST tagger (existing dep), OpenAI `whisper-1` + `gpt-4o-mini` (existing `openai` dep), soundfile, ffmpeg/streamlink (existing `probe` extra).

## Global Constraints

- **Run via `uv run`.** uv managed CPython only.
- **Tests flat in `tests/` as `test_<name>.py`; pytest default deselects `slow`.** Default-run tests use **injected fakes only** — no real models, no network, no ffmpeg. Real AST/Whisper/LLM paths are `@pytest.mark.slow`. Mirror `tests/test_radio.py` / `tests/test_probe_*` (inject `runner=`, `client=`, `scorer=`, `clean=`, `select=`, `verify=`).
- **No new runtime dependencies:** AST tagger uses `transformers`+`torchaudio` (core); Whisper + LLM use the `openai` client (core); structured output via `pydantic` (already used in `ingest/language_filter.py`). Podcast/HLS capture deps are the existing `probe` extra.
- **16 kHz mono throughout:** `TARGET_SAMPLE_RATE = 16_000` from `musiclang.config`.
- **Segment bar:** exactly one **≥30 s** clean-speech segment per recording (`SEGMENT_LENGTH_S = 30.0`).
- **8 seed languages, fixed:** keys of `musiclang.config.SEED_LANGUAGES` (english, german, polish, french, spanish, italian, greek, finnish). Whisper's `verbose_json.language` is a lowercase English name (e.g. `"english"`) — compares directly to these keys.
- **Verification labels:** `{target-speech | music | other-language | other}`. **Fail policy = drop-and-log:** keep only `target-speech`; drop the rest; log every drop per language + reason.
- **Budget:** **≥25 verified segments/language across ≥4 distinct channels**; keep collecting until met or the channel pool is exhausted (logged — never silently truncate).
- **Hard PILOT gate:** after the pipeline is built (Tasks 1–6), run 1 verified segment/language (Task 7), present verdicts, and **STOP for the user's manual green-light** before the full run (Task 8).
- **ffmpeg/streamlink are NOT on the tool-shell PATH** — prepend the winget ffmpeg bin for capture runs (`/c/Users/nikol/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_*/ffmpeg-*/bin`); streamlink resolves in the venv under `uv run`.
- **`data/` is gitignored** — segments/manifests/wavs stay local. **Branch:** `data-integrity-brief`. Commit after every code task.
- **Provenance (design spec §9):** cite Whisper (Radford et al. 2022, arXiv:2212.04356) and AST (Gong et al. 2021, arXiv:2104.01778) in docstrings.

## File Structure

| File | Responsibility |
|---|---|
| `src/musiclang/clean/select.py` | `select_segment` — first ≥30 s window of concatenated clean speech, else None. Pure. |
| `src/musiclang/verify/__init__.py` | Package docstring. |
| `src/musiclang/verify/verifier.py` | `Verdict` + `verify_segment` — the hybrid short-circuit over injectable tagger/whisper/llm. |
| `src/musiclang/verify/tagger.py` | `TaggerScores` + `tag_speech_music` (AST speech/music; injectable scorer). |
| `src/musiclang/verify/whisper_id.py` | `transcribe_language` (OpenAI Whisper → detected language + transcript). |
| `src/musiclang/verify/llm_judge.py` | `judge_transcript` (gpt-4o-mini fluency tie-break; pydantic structured output). |
| `src/musiclang/ingest/manifest.py` (modify) | add `SEGMENTS_COLUMNS`/`segments_manifest_dataframe` + `DROPS_COLUMNS`/`drops_dataframe`. |
| `scripts/collect_segments.py` | breadth collector: `process_recording` + per-language budget loop + `--pilot`/full + drop logging. |
| `tests/test_select.py`, `tests/test_verify_*.py`, `tests/test_segments_manifest.py`, `tests/test_collect_segments.py` | unit tests (fakes). |

---

### Task 1: `select_segment` — one 30 s window of clean speech

**Files:** Create `src/musiclang/clean/select.py`; Test `tests/test_select.py`.

**Interfaces:**
- Consumes: `window_signal` from `musiclang.clean.window`.
- Produces: `SEGMENT_LENGTH_S = 30.0`; `select_segment(signal, sr=TARGET_SAMPLE_RATE, length_s=SEGMENT_LENGTH_S) -> np.ndarray | None` — the first window of exactly `length_s` from `signal` (already-cleaned, concatenated speech), or `None` if `signal` is shorter than `length_s`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_select.py
import numpy as np
from musiclang.clean.select import select_segment, SEGMENT_LENGTH_S


def test_select_returns_first_30s_when_enough_speech():
    sr = 16_000
    signal = np.arange(sr * 45, dtype=np.float32)  # 45 s of clean speech
    seg = select_segment(signal, sr=sr)
    assert seg is not None
    assert len(seg) == int(SEGMENT_LENGTH_S * sr)
    assert seg[0] == 0.0 and seg[-1] == float(int(SEGMENT_LENGTH_S * sr) - 1)  # first window


def test_select_returns_none_when_under_30s():
    sr = 16_000
    signal = np.zeros(sr * 20, dtype=np.float32)  # only 20 s clean
    assert select_segment(signal, sr=sr) is None


def test_select_exactly_30s_is_accepted():
    sr = 16_000
    signal = np.zeros(int(SEGMENT_LENGTH_S * sr), dtype=np.float32)
    seg = select_segment(signal, sr=sr)
    assert seg is not None and len(seg) == int(SEGMENT_LENGTH_S * sr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_select.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'musiclang.clean.select'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/clean/select.py
"""Select ONE analysis segment per recording: the first fixed-length window of
the already-cleaned, concatenated speech. One segment per recording keeps
segments independent (no within-recording pseudoreplication)."""

from __future__ import annotations

import numpy as np

from musiclang.clean.window import window_signal
from musiclang.config import TARGET_SAMPLE_RATE

SEGMENT_LENGTH_S: float = 30.0


def select_segment(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, length_s: float = SEGMENT_LENGTH_S
) -> np.ndarray | None:
    """First `length_s`-second window of `signal`, or None if `signal` is shorter.

    `signal` is the cleaned, concatenated speech of one recording (see
    `pipeline.clean_clip`). Returns the first non-overlapping window so the
    choice is deterministic.
    """
    windows = window_signal(signal, sr, length_s)
    return windows[0].samples if windows else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_select.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/clean/select.py tests/test_select.py
git commit -m "feat(clean): select_segment — one 30s window of clean speech per recording"
```

---

### Task 2: `Verdict` + `verify_segment` hybrid orchestration

**Files:** Create `src/musiclang/verify/__init__.py`, `src/musiclang/verify/verifier.py`; Test `tests/test_verify_verifier.py`.

**Interfaces:**
- Produces:
  - `Verdict` dataclass: `label, confidence, detected_language, transcript, tagger_speech, tagger_music, stage_decided`.
  - `verify_segment(signal, sr, target_language, *, tagger=None, whisper=None, llm_judge=None, music_threshold=0.5, speech_threshold=0.3) -> Verdict`.
  - The three callables (injected; lazy-default to the real impls in Tasks 3–4): `tagger(signal, sr) -> TaggerScores(speech, music)`; `whisper(signal, sr) -> (detected_language, transcript)`; `llm_judge(transcript, target_language) -> bool`.
- **Logic (short-circuit):**
  1. `s = tagger(...)`. If `s.music >= music_threshold and s.music > s.speech` → `music` (stage `tagger`).
  2. `lang, text = whisper(...)`. If `lang == target_language` and `len(text) >= 3` → `target-speech` (stage `whisper`).
  3. else tie-break `fluent = llm_judge(text, target_language)`: `fluent` → `target-speech` (stage `llm`); elif `lang and lang != target_language` → `other-language` (stage `llm`); else → `other` (stage `llm`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify_verifier.py
import numpy as np
from musiclang.verify.verifier import Verdict, verify_segment
from musiclang.verify.tagger import TaggerScores

SIG = np.zeros(16_000, dtype=np.float32)


def _tagger(speech, music):
    return lambda sig, sr: TaggerScores(speech=speech, music=music)


def test_music_rejected_at_tagger_stage():
    v = verify_segment(SIG, 16_000, "english",
                       tagger=_tagger(0.1, 0.9),
                       whisper=lambda s, sr: ("english", "should not run"),
                       llm_judge=lambda t, l: True)
    assert v.label == "music" and v.stage_decided == "tagger"


def test_target_language_accepted_at_whisper_stage():
    v = verify_segment(SIG, 16_000, "german",
                       tagger=_tagger(0.9, 0.1),
                       whisper=lambda s, sr: ("german", "guten tag wie geht es"),
                       llm_judge=lambda t, l: (_ for _ in ()).throw(AssertionError("llm must not run")))
    assert v.label == "target-speech" and v.stage_decided == "whisper"
    assert v.detected_language == "german"


def test_language_mismatch_goes_to_tiebreak_then_other_language():
    v = verify_segment(SIG, 16_000, "english",
                       tagger=_tagger(0.8, 0.1),
                       whisper=lambda s, sr: ("french", "bonjour tout le monde"),
                       llm_judge=lambda t, l: False)
    assert v.label == "other-language" and v.stage_decided == "llm"


def test_tiebreak_rescues_target_speech_on_short_transcript():
    v = verify_segment(SIG, 16_000, "english",
                       tagger=_tagger(0.8, 0.1),
                       whisper=lambda s, sr: ("english", "hi"),  # too short -> tiebreak
                       llm_judge=lambda t, l: True)
    assert v.label == "target-speech" and v.stage_decided == "llm"


def test_empty_transcript_other():
    v = verify_segment(SIG, 16_000, "english",
                       tagger=_tagger(0.6, 0.2),
                       whisper=lambda s, sr: ("", ""),
                       llm_judge=lambda t, l: False)
    assert v.label == "other" and v.stage_decided == "llm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_verify_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'musiclang.verify'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/verify/__init__.py
"""Audio-based segment verification (Workstream C): hybrid AST speech/music
tagger -> Whisper language-ID -> light LLM tie-break. Supersedes the metadata
is_in_language guard with audio evidence."""
```

```python
# src/musiclang/verify/verifier.py
"""Hybrid segment verifier: tagger -> whisper -> LLM tie-break (short-circuit)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE

_MIN_TRANSCRIPT = 3  # chars; below this a Whisper transcript is treated as garbage -> tie-break


@dataclass(frozen=True)
class Verdict:
    label: str            # target-speech | music | other-language | other
    confidence: float
    detected_language: str
    transcript: str
    tagger_speech: float
    tagger_music: float
    stage_decided: str    # tagger | whisper | llm


def verify_segment(
    signal: np.ndarray,
    sr: int,
    target_language: str,
    *,
    tagger=None,
    whisper=None,
    llm_judge=None,
    music_threshold: float = 0.5,
    speech_threshold: float = 0.3,
) -> Verdict:
    """Label `signal` as target-speech / music / other-language / other.

    `tagger(signal, sr) -> TaggerScores`; `whisper(signal, sr) -> (language, transcript)`;
    `llm_judge(transcript, target_language) -> bool`. All injectable; lazy-default to the
    real implementations so unit tests stay offline.
    """
    if tagger is None:
        from musiclang.verify.tagger import tag_speech_music as tagger
    if whisper is None:
        from musiclang.verify.whisper_id import transcribe_language as whisper
    if llm_judge is None:
        from musiclang.verify.llm_judge import judge_transcript as llm_judge

    scores = tagger(signal, sr)
    if scores.music >= music_threshold and scores.music > scores.speech:
        return Verdict("music", scores.music, "", "", scores.speech, scores.music, "tagger")

    lang, text = whisper(signal, sr)
    lang = (lang or "").lower()
    text = (text or "").strip()

    if lang == target_language.lower() and len(text) >= _MIN_TRANSCRIPT:
        return Verdict("target-speech", scores.speech, lang, text,
                       scores.speech, scores.music, "whisper")

    fluent = llm_judge(text, target_language)
    if fluent:
        label = "target-speech"
    elif lang and lang != target_language.lower():
        label = "other-language"
    else:
        label = "other"
    return Verdict(label, scores.speech, lang, text, scores.speech, scores.music, "llm")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_verify_verifier.py -v`
Expected: 5 passed. (The lazy imports never fire — all tests inject the three callables.)

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/verify/__init__.py src/musiclang/verify/verifier.py tests/test_verify_verifier.py
git commit -m "feat(verify): Verdict + hybrid verify_segment orchestration (tagger/whisper/llm)"
```

---

### Task 3: AST speech/music tagger

**Files:** Create `src/musiclang/verify/tagger.py`; Test `tests/test_verify_tagger.py`.

**Interfaces:**
- Produces: `TaggerScores` dataclass `(speech: float, music: float)`; `tag_speech_music(signal, sr=TARGET_SAMPLE_RATE, *, scorer=None) -> TaggerScores` — `scorer(signal, sr) -> dict[label, prob]` injectable; default `_ast_scorer` (real AST, lazy, slow). `music = max(Music, Singing)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify_tagger.py
import numpy as np
from musiclang.verify.tagger import TaggerScores, tag_speech_music


def test_tag_extracts_speech_and_max_music_singing():
    fake = lambda sig, sr: {"Speech": 0.91, "Music": 0.10, "Singing": 0.40, "Dog": 0.02}
    s = tag_speech_music(np.zeros(16_000, dtype=np.float32), scorer=fake)
    assert isinstance(s, TaggerScores)
    assert s.speech == 0.91
    assert s.music == 0.40  # max(Music, Singing)


def test_tag_missing_labels_default_zero():
    s = tag_speech_music(np.zeros(10, dtype=np.float32), scorer=lambda sig, sr: {})
    assert s.speech == 0.0 and s.music == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_verify_tagger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'musiclang.verify.tagger'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/verify/tagger.py
"""Speech-vs-music tagging via an AudioSet AST classifier (Gong et al. 2021,
arXiv:2104.01778; AudioSet: Gemmeke et al. 2017). Catches sung vocals / music
that the VAD speech gate let through. The scorer is injectable so unit tests
never load the model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from musiclang.config import TARGET_SAMPLE_RATE

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
_AST_INPUT_S = 10.0  # AST is trained on ~10 s inputs; window the 30 s and average


@dataclass(frozen=True)
class TaggerScores:
    speech: float
    music: float


def tag_speech_music(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, *, scorer=None
) -> TaggerScores:
    """Speech and music probabilities for `signal`. `music = max(Music, Singing)`."""
    scorer = scorer or _ast_scorer
    probs = scorer(signal, sr)
    speech = float(probs.get("Speech", 0.0))
    music = max(float(probs.get("Music", 0.0)), float(probs.get("Singing", 0.0)))
    return TaggerScores(speech=speech, music=music)


@lru_cache(maxsize=1)
def _load_ast():
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    model = ASTForAudioClassification.from_pretrained(AST_MODEL)
    model.eval()
    fe = AutoFeatureExtractor.from_pretrained(AST_MODEL)
    return model, fe


def _ast_scorer(signal: np.ndarray, sr: int) -> dict[str, float]:
    """Mean per-class sigmoid probability over ~10 s windows of `signal`."""
    import torch

    model, fe = _load_ast()
    win = int(_AST_INPUT_S * sr)
    chunks = [signal[i : i + win] for i in range(0, max(len(signal), 1), win)]
    sums = None
    n = 0
    for ch in chunks:
        if len(ch) < sr:  # skip a <1 s tail
            continue
        inputs = fe(ch, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            probs = torch.sigmoid(model(**inputs).logits)[0]
        sums = probs if sums is None else sums + probs
        n += 1
    if n == 0:
        return {}
    mean = sums / n
    id2label = model.config.id2label
    return {id2label[i]: float(mean[i]) for i in range(mean.shape[0])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_verify_tagger.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/verify/tagger.py tests/test_verify_tagger.py
git commit -m "feat(verify): AST speech/music tagger (injectable scorer; real AST lazy)"
```

---

### Task 4: Whisper language-ID + LLM tie-break (OpenAI)

**Files:** Create `src/musiclang/verify/whisper_id.py`, `src/musiclang/verify/llm_judge.py`; Test `tests/test_verify_openai.py`.

**Interfaces:**
- Produces:
  - `transcribe_language(signal, sr=TARGET_SAMPLE_RATE, *, client=None) -> tuple[str, str]` — `(detected_language_lowercased, transcript)`; lazy `OpenAI()`; returns `("", "")` on any error (fail-soft → downstream tie-break).
  - `judge_transcript(transcript, target_language, *, client=None) -> bool` — fluent target-language? `gpt-4o-mini` structured output; `False` for empty/short transcript and **fail-closed** (False) on API error (a borderline segment that can't be verified is dropped — the budget loop just collects another).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify_openai.py
from types import SimpleNamespace

import numpy as np

from musiclang.verify.whisper_id import transcribe_language
from musiclang.verify.llm_judge import judge_transcript


class _FakeTranscriptions:
    def __init__(self, resp):
        self._resp = resp
    def create(self, **kwargs):
        return self._resp


class _FakeWhisperClient:
    def __init__(self, language, text):
        self.audio = SimpleNamespace(
            transcriptions=_FakeTranscriptions(SimpleNamespace(language=language, text=text))
        )


def test_transcribe_language_lowercases_and_strips(tmp_path):
    client = _FakeWhisperClient("English", "  hello world  ")
    lang, text = transcribe_language(np.zeros(16_000, dtype=np.float32), client=client)
    assert lang == "english" and text == "hello world"


def test_transcribe_language_failsoft_on_error():
    class Boom:
        audio = SimpleNamespace(transcriptions=SimpleNamespace(
            create=lambda **k: (_ for _ in ()).throw(RuntimeError("api down"))))
    lang, text = transcribe_language(np.zeros(16_000, dtype=np.float32), client=Boom())
    assert lang == "" and text == ""


class _FakeParsed:
    def __init__(self, value):
        self.choices = [SimpleNamespace(message=SimpleNamespace(
            parsed=SimpleNamespace(is_fluent_target_language=value, reason="x")))]


class _FakeChatClient:
    def __init__(self, value):
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            parse=lambda **k: _FakeParsed(value)))


def test_judge_true_for_fluent():
    assert judge_transcript("una larga frase en español clara", "spanish",
                            client=_FakeChatClient(True)) is True


def test_judge_false_for_short_transcript_without_calling_api():
    # empty/short -> False, must not need a client
    assert judge_transcript("", "english", client=None) is False
    assert judge_transcript("hi", "english", client=None) is False


def test_judge_failclosed_on_api_error():
    class Boom:
        chat = SimpleNamespace(completions=SimpleNamespace(
            parse=lambda **k: (_ for _ in ()).throw(RuntimeError("api down"))))
    assert judge_transcript("a long enough transcript here", "english", client=Boom()) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_verify_openai.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'musiclang.verify.whisper_id'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/musiclang/verify/whisper_id.py
"""OpenAI Whisper language-ID + transcript (Radford et al. 2022, arXiv:2212.04356).
verbose_json `.language` is a lowercase English name (e.g. 'english'), comparable
directly to the SEED_LANGUAGES keys. Fail-soft: ('', '') on error -> downstream tie-break."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from musiclang.config import TARGET_SAMPLE_RATE

WHISPER_MODEL = "whisper-1"


def transcribe_language(
    signal: np.ndarray, sr: int = TARGET_SAMPLE_RATE, *, client=None
) -> tuple[str, str]:
    """Return (detected_language_lowercased, transcript) for `signal`."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "seg.wav"
            sf.write(str(wav), signal, sr)
            with open(wav, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    model=WHISPER_MODEL, file=fh, response_format="verbose_json"
                )
        lang = (getattr(resp, "language", "") or "").lower()
        text = (getattr(resp, "text", "") or "").strip()
        return lang, text
    except Exception:
        return "", ""
```

```python
# src/musiclang/verify/llm_judge.py
"""Light gpt-4o-mini tie-break: is a transcript fluent speech in the target
language? Mirrors ingest.language_filter.is_in_language (pydantic structured
output, lazy client). Fail-CLOSED (False) on error: a borderline segment that
can't be verified is dropped; the collector simply gathers another."""

from __future__ import annotations

from pydantic import BaseModel

MODEL = "gpt-4o-mini"
_MIN_TRANSCRIPT = 3


class FluencyVerdict(BaseModel):
    is_fluent_target_language: bool
    reason: str


def judge_transcript(transcript: str, target_language: str, *, client=None) -> bool:
    """True iff `transcript` reads as fluent `target_language` speech."""
    transcript = (transcript or "").strip()
    if len(transcript) < _MIN_TRANSCRIPT:
        return False
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    try:
        resp = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "You judge whether a speech transcript is FLUENT, natural speech in a "
                    "target language. Set is_fluent_target_language=False for music lyrics, "
                    "noise, gibberish, or text clearly in a DIFFERENT language."
                )},
                {"role": "user", "content": (
                    f"Target language: {target_language}.\nTranscript: {transcript!r}\n"
                    "Is this fluent target-language speech?"
                )},
            ],
            response_format=FluencyVerdict,
        )
        verdict = resp.choices[0].message.parsed
        return bool(verdict and verdict.is_fluent_target_language)
    except Exception:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_verify_openai.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/verify/whisper_id.py src/musiclang/verify/llm_judge.py tests/test_verify_openai.py
git commit -m "feat(verify): Whisper language-ID + gpt-4o-mini transcript tie-break"
```

---

### Task 5: Segments manifest + drop log

**Files:** Modify `src/musiclang/ingest/manifest.py`; Test `tests/test_segments_manifest.py`.

**Interfaces:**
- Produces: `SEGMENTS_COLUMNS`, `segments_manifest_dataframe(rows) -> pd.DataFrame`; `DROPS_COLUMNS`, `drops_dataframe(rows) -> pd.DataFrame`. Canonical column order; missing keys → NaN (pandas default).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segments_manifest.py
from musiclang.ingest.manifest import (
    SEGMENTS_COLUMNS, segments_manifest_dataframe, DROPS_COLUMNS, drops_dataframe,
)


def test_segments_manifest_column_order_and_rows():
    rows = [{
        "segment_id": "english_bbc_ep1", "language": "english", "source": "podcast",
        "channel_id": "bbc-global-news", "recording_ref": "ep1", "recorded_at": "2026-06-30T10:00:00Z",
        "clean_speech_s": 78.6, "path": "data/segments/english/english_bbc_ep1.wav",
        "label": "target-speech", "confidence": 0.93, "detected_language": "english",
        "transcript": "hello", "tagger_speech": 0.9, "tagger_music": 0.1, "stage_decided": "whisper",
    }]
    df = segments_manifest_dataframe(rows)
    assert list(df.columns) == SEGMENTS_COLUMNS
    assert df.loc[0, "label"] == "target-speech" and df.loc[0, "language"] == "english"


def test_drops_dataframe_column_order():
    rows = [{"language": "greek", "source": "radio", "channel_id": "ert-deftero",
             "recording_ref": "cap1", "reason": "music", "detail": "tagger music=0.8"}]
    df = drops_dataframe(rows)
    assert list(df.columns) == DROPS_COLUMNS
    assert df.loc[0, "reason"] == "music"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segments_manifest.py -v`
Expected: FAIL — `ImportError: cannot import name 'SEGMENTS_COLUMNS'`.

- [ ] **Step 3: Write minimal implementation** (append to `src/musiclang/ingest/manifest.py`)

```python
SEGMENTS_COLUMNS = [
    "segment_id", "language", "source", "channel_id", "recording_ref", "recorded_at",
    "clean_speech_s", "path",
    "label", "confidence", "detected_language", "transcript",
    "tagger_speech", "tagger_music", "stage_decided",
]


def segments_manifest_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build the verified-segments manifest with the canonical column order."""
    return pd.DataFrame(rows, columns=SEGMENTS_COLUMNS)


DROPS_COLUMNS = ["language", "source", "channel_id", "recording_ref", "reason", "detail"]


def drops_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build the per-language drop log (why each candidate was rejected)."""
    return pd.DataFrame(rows, columns=DROPS_COLUMNS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_segments_manifest.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/musiclang/ingest/manifest.py tests/test_segments_manifest.py
git commit -m "feat(ingest): verified-segments manifest + drop-log dataframes"
```

---

### Task 6: Breadth collector

**Files:** Create `scripts/collect_segments.py`; Test `tests/test_collect_segments.py`.

**Interfaces:**
- Consumes: `probe.adapters` capture fns + `latest_enclosures`; `pipeline.clean_clip`; `clean.select.select_segment`; `verify.verifier.verify_segment` + `Verdict`; `ingest.manifest` (segments + drops); `config.SEED_LANGUAGES`/`DATA_DIR`/`TARGET_SAMPLE_RATE`.
- Produces (module-level, importable for tests):
  - `process_recording(wav_path, *, language, source, channel_id, recording_ref, recorded_at, clean=clean_clip, select=select_segment, verify=verify_segment, sr=TARGET_SAMPLE_RATE) -> tuple` — returns `("kept", segment_meta: dict, samples: np.ndarray)` or `("drop", reason: str, detail: str)`. Never raises (a processing error → `("drop", "error", <msg>)`).
  - `run(per_language: int, sources, pilot: bool, instances_path, out_dir) -> tuple[pd.DataFrame, pd.DataFrame]` — the collection loop (segments_df, drops_df); writes wavs + `segments_manifest.parquet` + `segments_drops.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_segments.py
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "collect_segments", Path(__file__).resolve().parents[1] / "scripts" / "collect_segments.py"
)
collect_segments = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect_segments)

from musiclang.verify.verifier import Verdict  # noqa: E402

_META = dict(language="english", source="podcast", channel_id="bbc",
             recording_ref="ep1", recorded_at="2026-06-30T10:00:00Z")


def _verdict(label):
    return Verdict(label, 0.9, "english", "hello world here", 0.9, 0.1, "whisper")


def test_process_recording_keeps_verified_target_speech(tmp_path):
    wav = tmp_path / "r.wav"; wav.write_bytes(b"x")
    status, meta, samples = collect_segments.process_recording(
        wav, **_META,
        clean=lambda p: np.zeros(16_000 * 40, dtype=np.float32),
        select=lambda sig, sr: np.zeros(16_000 * 30, dtype=np.float32),
        verify=lambda seg, sr, lang: _verdict("target-speech"),
    )
    assert status == "kept"
    assert meta["language"] == "english" and meta["label"] == "target-speech"
    assert meta["clean_speech_s"] == 40.0 and len(samples) == 16_000 * 30


def test_process_recording_drops_when_under_30s(tmp_path):
    wav = tmp_path / "r.wav"; wav.write_bytes(b"x")
    status, reason, detail = collect_segments.process_recording(
        wav, **_META,
        clean=lambda p: np.zeros(16_000 * 10, dtype=np.float32),
        select=lambda sig, sr: None,
        verify=lambda seg, sr, lang: _verdict("target-speech"),
    )
    assert status == "drop" and reason == "no-30s"


def test_process_recording_drops_on_verdict(tmp_path):
    wav = tmp_path / "r.wav"; wav.write_bytes(b"x")
    status, reason, detail = collect_segments.process_recording(
        wav, **_META,
        clean=lambda p: np.zeros(16_000 * 40, dtype=np.float32),
        select=lambda sig, sr: np.zeros(16_000 * 30, dtype=np.float32),
        verify=lambda seg, sr, lang: _verdict("music"),
    )
    assert status == "drop" and reason == "music"


def test_process_recording_never_raises(tmp_path):
    wav = tmp_path / "r.wav"; wav.write_bytes(b"x")
    status, reason, detail = collect_segments.process_recording(
        wav, **_META, clean=lambda p: (_ for _ in ()).throw(RuntimeError("bad audio")))
    assert status == "drop" and reason == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_collect_segments.py -v`
Expected: FAIL — `scripts/collect_segments.py` does not exist (spec load raises `FileNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/collect_segments.py
"""Collect ONE verified independent 30s clean-speech segment per recording, per
language, from podcasts (primary) + radio (supplement). Workstream B+C.

  uv run python scripts/collect_segments.py --pilot           # 1 segment/language
  uv run python scripts/collect_segments.py --per-language 25 # full run

Requires ffmpeg + streamlink on PATH (see the plan Global Constraints) and
OPENAI_API_KEY in .env. `data/` is gitignored.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import soundfile as sf

from musiclang.clean.select import select_segment
from musiclang.config import DATA_DIR, SEED_LANGUAGES, TARGET_SAMPLE_RATE
from musiclang.ingest.manifest import (
    DROPS_COLUMNS, SEGMENTS_COLUMNS, drops_dataframe, segments_manifest_dataframe,
)
from musiclang.pipeline import clean_clip
from musiclang.probe import adapters
from musiclang.probe.core import RecordingRef
from musiclang.verify.verifier import verify_segment


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")[:48]


def process_recording(
    wav_path, *, language, source, channel_id, recording_ref, recorded_at,
    clean=clean_clip, select=select_segment, verify=verify_segment, sr=TARGET_SAMPLE_RATE,
):
    """Clean -> select 30s -> verify one recording. Returns ('kept', meta, samples) or ('drop', reason, detail)."""
    try:
        signal = clean(wav_path)
        if len(signal) == 0:
            return ("drop", "no-30s", "empty after VAD")
        seg = select(signal, sr)
        if seg is None:
            return ("drop", "no-30s", f"{len(signal)/sr:.1f}s clean < 30s")
        v = verify(seg, sr, language)
        if v.label != "target-speech":
            detail = v.detected_language or f"music={v.tagger_music:.2f}"
            return ("drop", v.label, f"{v.stage_decided}:{detail}")
        meta = {
            "segment_id": f"{language}_{_slug(channel_id)}_{_slug(recording_ref)}",
            "language": language, "source": source, "channel_id": channel_id,
            "recording_ref": recording_ref, "recorded_at": recorded_at,
            "clean_speech_s": float(len(signal) / sr), "path": "",
            "label": v.label, "confidence": v.confidence, "detected_language": v.detected_language,
            "transcript": v.transcript[:500], "tagger_speech": v.tagger_speech,
            "tagger_music": v.tagger_music, "stage_decided": v.stage_decided,
        }
        return ("kept", meta, seg)
    except Exception as exc:  # noqa: BLE001 — a bad recording is a drop, never a crash
        return ("drop", "error", str(exc)[:200])


def _channels_for(language: str, instances: pd.DataFrame, sources) -> list[dict]:
    """Per-language channel list from the Workstream-A instances (podcast feeds + radio)."""
    df = instances[(instances["language"] == language) & (instances["source"].isin(sources))]
    return df.to_dict("records")


def _recordings(channel: dict, per_channel: int):
    """Yield (recording_ref, capture_kind, capture_arg) for a channel."""
    if channel["kind"] == "rss_feed":
        for i, url in enumerate(adapters.latest_enclosures(channel["ref"], per_channel)):
            yield (f"{_slug(channel['channel_id'])}-ep{i}", "rss", url)
    else:  # hls | progressive  -> one capture per station
        yield (_slug(channel["channel_id"]), channel["kind"], channel["ref"])


def run(per_language=25, sources=("podcast", "radio"), pilot=False,
        instances_path=None, out_dir=None):
    instances_path = Path(instances_path or DATA_DIR / "source_instances.parquet")
    out_dir = Path(out_dir or DATA_DIR / "segments")
    work = DATA_DIR / "_seg_work"; work.mkdir(parents=True, exist_ok=True)
    instances = pd.read_parquet(instances_path)
    target = 1 if pilot else per_language
    per_channel = 1 if pilot else max(3, per_language // 2)

    seg_rows, drop_rows = [], []
    for language in SEED_LANGUAGES:
        kept = 0
        for ch in _channels_for(language, instances, sources):
            if kept >= target:
                break
            for rec_ref, kind, arg in _recordings(ch, per_channel):
                if kept >= target:
                    break
                ref = RecordingRef(ch["source"], language, ch["channel_id"], kind, arg)
                wav = work / f"{language}_{rec_ref}.wav"
                if adapters.CAPTURE_DISPATCH[kind](ref, wav) is None:
                    drop_rows.append({"language": language, "source": ch["source"],
                                      "channel_id": ch["channel_id"], "recording_ref": rec_ref,
                                      "reason": "capture-failed", "detail": kind})
                    continue
                recorded_at = datetime.now(timezone.utc).isoformat()
                status, a, b = process_recording(
                    wav, language=language, source=ch["source"], channel_id=ch["channel_id"],
                    recording_ref=rec_ref, recorded_at=recorded_at)
                if status == "kept":
                    seg_path = out_dir / language / f"{a['segment_id']}.wav"
                    seg_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(seg_path), b, TARGET_SAMPLE_RATE)
                    a["path"] = str(seg_path)
                    seg_rows.append(a)
                    kept += 1
                    print(f"[keep] {language} {a['channel_id']} {rec_ref} ({kept}/{target})")
                else:
                    drop_rows.append({"language": language, "source": ch["source"],
                                      "channel_id": ch["channel_id"], "recording_ref": rec_ref,
                                      "reason": a, "detail": b})
                    print(f"[drop:{a}] {language} {ch['channel_id']} {rec_ref}")
        print(f"=== {language}: {kept}/{target} verified ===")

    seg_df = segments_manifest_dataframe(seg_rows)
    drop_df = drops_dataframe(drop_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_pilot" if pilot else ""
    seg_df.to_parquet(DATA_DIR / f"segments_manifest{suffix}.parquet")
    drop_df.to_parquet(DATA_DIR / f"segments_drops{suffix}.parquet")
    print(f"\nWROTE {len(seg_df)} segments, {len(drop_df)} drops "
          f"({DATA_DIR / f'segments_manifest{suffix}.parquet'})")
    return seg_df, drop_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect verified independent 30s speech segments.")
    p.add_argument("--pilot", action="store_true", help="1 verified segment/language (review gate)")
    p.add_argument("--per-language", type=int, default=25)
    p.add_argument("--sources", type=str, default="podcast,radio")
    p.add_argument("--instances", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    run(per_language=args.per_language,
        sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
        pilot=args.pilot, instances_path=args.instances, out_dir=args.out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_collect_segments.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/collect_segments.py tests/test_collect_segments.py
git commit -m "feat(collect): breadth segment collector (capture -> clean -> select -> verify -> manifest)"
```

---

### Task 7: PILOT run (1 verified segment/language) — HARD user-review gate

**Files:** Produces `data/segments_manifest_pilot.parquet`, `data/segments_drops_pilot.parquet`, `data/segments/<lang>/*.wav` (gitignored). No new code.

Controller-driven (real network + models). **This task ends by STOPPING for the user's manual review — do NOT proceed to Task 8 without an explicit green-light.**

- [ ] **Step 1: Preconditions** — `data/source_instances.parquet` exists (Workstream A) and `OPENAI_API_KEY` is in `.env`.

Run: `uv run python -c "import pandas as pd,os; from dotenv import load_dotenv; load_dotenv(); print('instances:', len(pd.read_parquet('data/source_instances.parquet'))); print('key:', bool(os.getenv('OPENAI_API_KEY')))"`
Expected: instances > 0 and `key: True`.

- [ ] **Step 2: Warm up / sanity-check the AST tagger once** (first load downloads the model)

Run: `uv run python -c "import numpy as np; from musiclang.verify.tagger import tag_speech_music; print(tag_speech_music(np.random.randn(16000*5).astype('float32')))"`
Expected: a `TaggerScores(speech=..., music=...)` printed (confirms the model loads + runs on CPU).

- [ ] **Step 3: Run the pilot** (ffmpeg on PATH; background controller job)

```bash
export PATH="/c/Users/nikol/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin:$PATH"
uv run python -u scripts/collect_segments.py --pilot
```
Run with `run_in_background: true`; monitor the `[keep]/[drop]` progress.

- [ ] **Step 4: Present the pilot verdicts for manual review**

Run: `uv run python -c "import pandas as pd; pd.set_option('display.max_colwidth',80); df=pd.read_parquet('data/segments_manifest_pilot.parquet'); print(df[['language','source','channel_id','label','confidence','detected_language','tagger_speech','tagger_music','stage_decided','path']].to_string(index=False)); print(); [print(r.language,'::',r.transcript[:160]) for r in df.itertuples()]"`

Then present to the user, per segment: language · source · channel · label · confidence · detected_language · tagger scores · stage · **wav path (listenable)** · transcript snippet. Also show any pilot drops (`data/segments_drops_pilot.parquet`).

- [ ] **Step 5: STOP — await the user's green-light.** Ask the user to listen to the wavs and confirm the C verdicts look right. If they flag problems (mis-labels, thresholds), tune (`music_threshold`/`speech_threshold` in `verify_segment`, or prompts) and re-pilot. **Only proceed to Task 8 on explicit approval.**

---

### Task 8: FULL collection (≥25 verified segments/language)

**Files:** Produces `data/segments_manifest.parquet`, `data/segments_drops.parquet`, `data/segments/<lang>/*.wav` (gitignored). No new code. Controller-driven background job, **only after Task 7 green-light.**

- [ ] **Step 1: Launch the full run** (ffmpeg on PATH)

```bash
export PATH="/c/Users/nikol/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin:$PATH"
uv run python -u scripts/collect_segments.py --per-language 25
```
Run with `run_in_background: true`; monitor. Podcast episode downloads + Whisper calls dominate wall-clock.

- [ ] **Step 2: Report coverage + drops**

Run: `uv run python -c "import pandas as pd; s=pd.read_parquet('data/segments_manifest.parquet'); d=pd.read_parquet('data/segments_drops.parquet'); print('per-language verified:'); print(s.groupby('language').agg(n=('segment_id','size'), channels=('channel_id','nunique'))); print('\ndrops by language x reason:'); print(d.groupby(['language','reason']).size())"`
Expected: per-language verified counts (target ≥25, ≥4 channels) + the drop breakdown. **Honestly report any language under target** (don't silently truncate); for a short language, fall back to its working radio + more episodes and re-run that language, or record the shortfall for the D+E findings.

- [ ] **Step 3: No commit** (dataset is gitignored). The verified-segment dataset now feeds Workstream D+E.

---

## Self-Review

**Spec coverage** (spec section → task):
- §2.2 one 30 s segment/recording → Task 1 (`select_segment`). ✓
- §5 hybrid verification (tagger→whisper→LLM, short-circuit) → Task 2 (orchestration) + Task 3 (tagger) + Task 4 (whisper+LLM). ✓
- §5 labels + drop-and-log → `Verdict.label` (Task 2) + drop log (Task 5) + collector drop rows (Task 6). ✓
- §4 collection flow (pool→clean→select→verify→budget), sources, one-per-recording → Task 6 `run`/`process_recording`. ✓
- §6 pilot checkpoint + manual review gate → Task 7 (hard STOP). ✓
- §7 artifacts (segment wavs + manifest + drop log) → Task 5 (schemas) + Task 6 (writes) + Tasks 7/8 (runs). ✓
- §8 reuse (probe adapters, clean_clip, window, manifest, openai pattern) → Tasks 1/6 reuse; §8 new modules → Tasks 1–6 file map. ✓
- §9 testing (fakes default, slow real) → every code task injects fakes; real AST/Whisper are runtime-only (Task 7 warms them). ✓
- §10 no new deps + citations → constraints + docstrings (Tasks 3/4). ✓
- §11 risks (tagger tuning via pilot; coverage shortfall logged; fail-closed tiebreak) → Task 7 tuning loop + Task 8 honest reporting + Task 4 fail-closed. ✓
- §12 success (≥25/lang, pilot-approved) → Tasks 7+8. ✓

**Placeholder scan:** no "TBD"/"add error handling"/"similar to Task N" — every code+test block is complete. ✓

**Type consistency:** `Verdict(label, confidence, detected_language, transcript, tagger_speech, tagger_music, stage_decided)` identical in Tasks 2/6; `TaggerScores(speech, music)` in Tasks 2/3; `verify_segment(signal, sr, target_language, *, tagger, whisper, llm_judge, ...)` matches its lazy-default imports (Tasks 3/4) and its call in `process_recording` (Task 6); `transcribe_language(signal, sr, *, client)` and `judge_transcript(transcript, target_language, *, client)` match the verifier's lazy imports; `SEGMENTS_COLUMNS`/`DROPS_COLUMNS` (Task 5) match the dicts built in Task 6; `CAPTURE_DISPATCH[kind]` keys (`rss`/`hls`/`progressive`) match `_recordings`' emitted kinds. ✓

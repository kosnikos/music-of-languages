# Workstream B+C — Verified Independent-Segment Dataset — Design Spec

**Phase:** Data Integrity (0.75) · **Workstreams:** B (independent 30 s segments) + C (LLM verification)
**Date:** 2026-06-30 · **Branch:** `data-integrity-brief`

Covers **B+C together** as one "verified-segment dataset" pipeline (user decision: collection and
verification are coupled). **D+E (outlier detection, robust aggregation, the station-vs-language
confound re-test, interim clustering, `docs/data-integrity-findings.md`) get their own brainstorm →
spec → plan cycle next.** Builds directly on Workstream A's findings
(`docs/source-evaluation.md`): podcasts primary, radio supplement, corpus dropped, budget ≥25
segments/language across ≥4 channels. Companion: `docs/data-integrity-phase.md` (Workstreams B & C),
`docs/superpowers/specs/2026-06-24-music-of-languages-design.md` (§7 robustness, §9 conventions).

---

## 1. Summary

Produce, for each of the 8 seed languages, a set of **independent ~30 s clean-speech segments —
exactly one per recording, no two sharing a recording or channel** — each **audio-verified** as
fluent target-language speech and labelled `{target-speech | music | other-language | other}` with
confidence. The output is a **trustworthy audio dataset + provenance manifest + per-segment verdicts**
that Workstream D+E then analyzes. Verification is a **hybrid gate** (local speech/music tagger →
Whisper language-ID → light LLM tie-break), superseding the metadata `is_in_language` guard with
audio evidence. A **pilot checkpoint** (1 verified segment/language) is manually reviewed by the user
before the full collection runs.

**Seed languages (8, fixed):** english, german, polish, french, spanish, italian, greek, finnish.

## 2. Decisions locked in the brainstorm

1. **Chunking:** B+C together now; D+E next. (Collection isn't trustworthy until verified.)
2. **Analysis unit (phase-brief user directive):** **one ~30 s clean-speech segment per recording.**
   Segments are independent of one another → kills within-recording pseudoreplication.
3. **Sources (Workstream A):** podcasts **primary** (1 episode = 1 independent recording),
   radio **supplement** (radio-browser pool + HLS), **German is podcast-only** (live radio
   geo-blocked from this host). Corpus **dropped** (VoxPopuli non-native pollution).
4. **Selection rule:** the **first 30 s window of the concatenated clean speech** (deterministic;
   reuses `clean_clip` + `window_signal`). Not centered/longest-run — simplicity over marginal gain.
5. **Verification (C) = hybrid**, drop-and-log fail policy (below).
6. **Budget:** **≥25 verified segments/language across ≥4 distinct channels.** Reached via podcast
   episodes (each an independent recording at a distinct time) → no hourly radio scheduler needed now.
7. **Pilot checkpoint before full collection** (user-added requirement): 1 verified segment/language,
   manually reviewed.

## 3. Goals & non-goals

**Goals**
- A **breadth collector** that gathers many independent recordings/language across many channels
  from the Workstream-A sources, producing **one ≥30 s clean-speech segment per recording**.
- A **hybrid verification gate** that labels each segment from audio evidence and drops non-target.
- A **verified-segment dataset** (wavs + provenance manifest + verdicts) meeting the budget, with
  per-language **drop/coverage logging**.
- A **pilot** (1/language) gated on the user's manual review of the C verdicts.

**Non-goals (B+C)**
- Feature extraction (prosody scalars, XLS-R embeddings + the slow-embedding cache) — **D+E**.
- Outlier detection / the swappable Validator ABC — **D**.
- Robust aggregation, the confound re-test, interim clustering, the findings doc — **E**.
- The hourly timed-radio scheduler (optional future depth; podcasts carry the budget).
- Production scheduling/retention hardening — Phase 1.

## 4. Collection (B)

**Flow (per language): collect a pool → clean + select → verify → keep to budget**, iterating until
≥25 verified or the channel pool is exhausted (logged).

- **Channels:** podcast feeds (from `data/source_instances.parquet` / the A research) + radio
  stations (those + live `ingest/radio.find_stations` enumeration). Aim ≥4 distinct channels/language.
- **Recordings per channel:** podcasts — pull several recent **episodes** per feed (each an
  independent recording); radio — capture per station (optionally at spaced times; not required now).
- **One segment per recording:** `clean_clip(recording)` (loudness-norm → silero VAD → concatenate
  clean speech); **require ≥30 s clean speech**; take the **first 30 s window** of the concatenated
  clean speech (`window_signal(..., length_s=30)[0]`). Recordings with <30 s clean speech are
  dropped (logged).
- **Independence invariant:** at most one kept segment per `recording_ref`; channels may contribute
  multiple recordings (different episodes/times), which remain independent.

## 5. Verification (C) — hybrid gate

Applied to each candidate 30 s segment, short-circuiting in order:

1. **Local speech/music tagger** — an AudioSet **AST** classifier via the existing `transformers`
   (default `MIT/ast-finetuned-audioset-10-10-0.4593`, CPU; the 30 s windowed into the model's input
   length, Speech vs Music/Singing scores averaged). If music/singing dominates (score thresholds) →
   **`music`**, reject. Behind a swappable `tagger(wav) -> {speech,music,...scores}` interface.
2. **Whisper language-ID** — OpenAI `whisper-1` (`response_format="verbose_json"` → detected
   `language` + `transcript`), reusing `OPENAI_API_KEY` from `.env`. If detected language ≠ target
   (confidently) → **`other-language`**, reject. (Local faster-whisper is the documented fallback.)
3. **LLM tie-break** — only on borderline cases (tagger uncertain; Whisper language-mismatch at low
   confidence; or an empty/too-short/garbage transcript): `gpt-4o-mini` (structured output) judges
   the transcript "fluent target-language speech vs foreign/garbage" — reusing the `is_in_language`
   pattern. Resolves to **`target-speech`** or **`other`**.

**Verdict** stored per segment: `label ∈ {target-speech, music, other-language, other}`, `confidence`,
`detected_language`, `transcript`, `tagger_scores`, `stage_decided ∈ {tagger, whisper, llm}`.

**Fail policy: drop-and-log.** Keep only `target-speech`; drop the rest; **log drops per language and
per reason** so coverage loss is visible. Keep collecting until the budget is met or the pool empties.

## 6. Pilot checkpoint (hard gate before full collection)

After the pipeline is built and unit-tested, run a **pilot: collect and verify ONE segment per
language (8 total)** end-to-end. Present to the user, per segment:
`language · source · channel_id · path (listenable wav) · label · confidence · detected_language ·
tagger_scores · transcript (first ~200 chars) · stage_decided`. **The user manually reviews the C
verdicts (the central concern) and explicitly green-lights** before the full ≥25/language run. If the
pilot reveals a verification problem (e.g. mis-labels, bad thresholds), tune and re-pilot.

## 7. Data flow & artifacts

- `data/segments/<lang>/<segment_id>.wav` — the 30 s clean segments (gitignored).
- **Segments manifest** `data/segments_manifest.parquet` (extends `ingest/manifest.py`):
  `segment_id, language, source (podcast|radio), channel_id, recording_ref, recorded_at,
  clean_speech_s, path` + verdict columns (`label, confidence, detected_language, transcript,
  tagger_speech, tagger_music, stage_decided`).
- **Drop log** `data/segments_drops.parquet`: per dropped candidate — `language, channel_id,
  recording_ref, reason (no-30s | music | other-language | other), detail`.

## 8. Components & interfaces (reuse, don't rebuild)

- **Reuse:** `probe/adapters.py` capture fns (graduate throwaway → collector use), `clean_clip` /
  `window_signal` / `concat_speech` (`clean/`), `ingest/manifest.py` + `ingest/radio.find_stations`,
  the OpenAI client pattern from `scripts/collect_sample.py`.
- **New — verification** `src/musiclang/verify/`:
  - `tagger.py` — `tag_speech_music(signal, sr) -> TaggerScores` (AST; injectable model).
  - `whisper_id.py` — `transcribe_language(wav, *, client) -> (language, transcript)`.
  - `verifier.py` — `verify_segment(signal, sr, target_language, *, tagger, whisper, llm) -> Verdict`
    (the hybrid short-circuit; all three stages injectable for tests).
- **New — collection**:
  - `clean/select.py` — `select_segment(signal, sr, length_s=30.0) -> np.ndarray | None`
    (first ≥30 s window of concatenated clean speech, else None).
  - `scripts/collect_segments.py` — the breadth collector + pilot/full modes + drop logging
    (controller-driven background run).
- Keep each file single-responsibility; the tagger, Whisper, and LLM are each swappable.

## 9. Testing

TDD with injected fakes (mirroring the probe harness):
- **Unit (default, offline):** fake capture/loader (collector); `select_segment` on synthetic
  signals; `verify_segment` with fake tagger-scores / fake Whisper response / fake LLM — covering each
  short-circuit branch (music→reject, other-language→reject, tiebreak→target/other) with no real
  models or network. Fakes only; mirror `tests/test_radio.py` / `tests/test_probe_*`.
- **Slow (integration, deselected):** the real AST tagger and real Whisper call marked
  `@pytest.mark.slow`.
- The pilot and full collection are **controller-driven background runs**, not tests.

## 10. Provenance & conventions (design spec §9)

Cite the verification models in docstrings: **Whisper** (Radford et al. 2022, arXiv:2212.04356) and
the **AST** tagger (Gong et al. 2021, arXiv:2104.01778; AudioSet: Gemmeke et al. 2017). **No new
runtime dependencies:** the AST tagger reuses `transformers` (core) + `torchaudio` (core); Whisper
reuses the `openai` client (core). (`faster-whisper`, the documented local fallback, would be an
optional `verify` extra only if ever needed; the podcast/HLS capture deps live in the existing `probe`
extra.) Run via `uv run`. ffmpeg stays off the tool-shell PATH (prepend the winget bin for capture
runs). 16 kHz mono throughout (`TARGET_SAMPLE_RATE`).

## 11. Risks & mitigations

- **Tagger over/under-rejects** → tunable thresholds; the pilot checkpoint surfaces this on real audio
  before the full run; the LLM tie-break backstops borderline cases.
- **Whisper mis-IDs short/garbled clips** → it runs on a 30 s clean segment (not a fragment); the
  tie-break catches low-confidence mismatches.
- **Coverage shortfall** for a thin language (e.g. Polish podcasts) → drop-and-log makes it visible;
  fall back to that language's working radio (PR24/TOK FM) + more episodes; report any language that
  can't reach ≥25 honestly (no silent truncation).
- **External-service exposure** (sending audio to OpenAI Whisper) → data is public radio/podcast; no
  privacy issue; documented.
- **German podcast-only** → ensure ≥4 German feeds yield enough episodes for ≥25; logged if short.

## 12. Success criteria

- **Primary:** a verified-segment dataset of **≥25 independent, audio-verified target-speech segments
  for as many of the 8 languages as the sources allow**, one per recording, across ≥4 channels, with
  per-language drop/coverage logging — and a **pilot the user approved** before the full run.
- **Secondary:** the verification gate demonstrably drops music/other-language on the pilot (the user
  confirms verdict quality manually).

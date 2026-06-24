# Phase 1 Handoff Notes — The Music of Languages

**Purpose:** capture decisions and rationale from the project kickoff (2026-06-24) so the agent
that plans **Phase 1** — *after* the Feature Exploration cycle (Phase 0.5) — doesn't re-derive them.
Companion to the design spec: `docs/superpowers/specs/2026-06-24-music-of-languages-design.md`.

This is intentionally light. Read the spec for full context; this doc only records the Phase-1-
relevant conclusions from the kickoff conversation.

## Where Phase 1 picks up

**Phase 0** stands up the pipeline + harness and validates ONE baseline method (alignment-free
prosody). The **Feature Exploration cycle (Phase 0.5)** then implements & compares additional
methods and **decides which feature method(s) to carry forward** (see
`docs/feature-exploration-cycle.md`). **Phase 1** hardens that decision into the end-to-end
pipeline and runs it on real radio for the seed languages
(catalog → ingest → clean → features → proximity → validate → viz).

## Key insight: only the `features` stage is gated by Phase 0

Most of Phase 1 does **not** depend on the Phase 0 outcome:

| Stage | Gated by Phase 0's method choice? |
|---|---|
| `catalog` (station selection) | No — method-agnostic |
| `ingest` (record clips) | No — but see "method ripple" below |
| `clean` (VAD, music/speech, normalize) | Mostly no — ripple only if a method needs special prep |
| `features` | **Yes — the only truly gated stage** |
| `proximity` (distance, clustering, MDS/UMAP) | No — consumes any feature vector |
| `validate` | Partly — approach(es) chosen in Phase 0 |
| `viz` (maps, dendrograms) | No — method-agnostic |

So Phase 1 can build/scale catalog, ingest, clean, proximity, and viz largely independent of the
feature-method decision; only `features` finalization waits for it (made in the Feature Exploration cycle).

## The "method ripple" (why Phase 1 wasn't finalized up front)

The three candidate feature methods differ **wildly** in infrastructure weight, which can reshape
`ingest`/`clean`. Read the Phase 0 outcome before finalizing those:

- **Alignment-free acoustic** (DisVoice / De Jong-Wempe / Parselmouth): **light**. `clean → features`
  is straightforward.
- **Forced-alignment metrics** (Montreal Forced Aligner + Correlatore-style battery): **heavy** —
  needs **per-language acoustic models**, possibly transcripts; reshapes `ingest`/`clean`.
- **Learned envelope embedding** (Deloche-style net): needs **model training**; data prep is
  amplitude envelopes + voicing markers.

## Phase 0 leaves you prototype helpers to HARDEN, not rewrite

To test methods on realistic data, Phase 0 builds **lightweight, portable** helpers for data
ingestion + cleaning: fetch a few real-radio clips per seed language; VAD + basic music/speech
filter + normalize. They use the **same stage boundaries/signatures** Phase 1 will use. Phase 1
should **harden and scale them in place**:

- **ingest:** multi-station / multi-time sampling, scheduling + retries, scale to ~30–40 languages.
- **clean:** speaker diarization (avoid single-speaker bias), better music/ad/jingle removal, robust
  normalization, quality gating.

## Phase 1-specific components deferred from Phase 0

- **Storage abstraction with retention policy** (`permanent` | `ephemeral`) — spec §6. Phase 0 just
  stores locally.
- **Per-language sample budget tuning** (stations × clips × hours) to stabilize metrics against the
  Arvaniti instability.
- **Confound controls validated end-to-end** (channel/codec, speaker diversity, speaking style).
- **proximity + viz hardened:** distance metric appropriate to the chosen feature type; the colored
  map (region color derived from the 2D proximity embedding).

## Carry-over decisions (do not relitigate)

- Prosody-first, modular `FeatureExtractor`; other verticals (SSL embeddings, phonetic inventory)
  combine in later.
- Treat proximity as a **continuous** space, not discrete rhythm classes.
- **Validation is a pluggable research area** — candidates: aggregate-over-variation, validate-
  against-reference-trees (family tree / typology / ASJP), outlier handling, others TBD. None
  hard-wired.
- Seed languages: English, German, Polish, French, Spanish, Italian, Greek, Finnish.
- Legal/ToS/copyright deferred to pre-publication.

## Open questions for Phase 1 (resolve after the Feature Exploration cycle)

- Which method(s) won — single or combined? → dictates `features` plus `clean`/`ingest` weight.
- Distance metric + clustering choices appropriate to the chosen feature type.
- Reference datasets for validation (which family-tree / typology / lexical sources).
- Sample budget per language to stabilize metrics.

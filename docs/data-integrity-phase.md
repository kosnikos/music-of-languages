# Data Integrity Phase (Phase 0.75) — Brief

**Purpose:** a dedicated **data-integrity** round that runs **after** the Feature Exploration
cycle (Phase 0.5; see `docs/feature-exploration-findings.md`) and **before** Phase 1. Phase 0.5
proved the comparison harness works but is **data- and confound-limited** — the SSL "win" over the
prosody baseline was weak and the segment-level geometry clustered **more by recording station than
by language**. This phase makes the per-language data **trustworthy** so the method comparison and
the first real language-proximity map mean something.

This is a **brief/outline only**. When the phase starts it gets its own full
**brainstorm → spec → plan → execute** pass (the next session picks up here). It deliberately
records candidate approaches + a recommended lean and leaves the binding choices to that brainstorm.

## Why this exists (what Phase 0.5 showed)

From `docs/feature-exploration-findings.md`:
- **Channel/station confound:** on the 55-clip corpus, segment-level cosine geometry separated by
  **station** (within/between gap **0.049**) *more* than by **language** (gap **0.012**). Multiple
  windows from one recording are correlated (same speaker/channel/codec) → apparent class signal is
  partly channel artefact and the effective N is inflated (pseudoreplication).
- **Thin/uneven coverage:** french/greek/spanish well-sampled; italian = 1 clip, finnish/german/
  english = 2–3. Per-language estimates are high-variance.
- **Metadata language-ID has a ceiling** (already documented in `docs/capital-stations-findings.md`):
  radio-browser tags mislabel BBC foreign-language services as `english`; music/jingles leak past
  the VAD gate. The robust fix is **audio-based** verification.

The fix is not another feature method — it is **data integrity**: independent units, audio-verified
content, outlier removal, and robust aggregation, then re-measure.

## Goals & non-goals

**Goals**
- Make each per-language sample a set of **independent, verified, clean 30 s speech segments**.
- Build the **outlier-detection + robust-validation** components (the swappable robustness strand
  deferred from Phase 0.5).
- Re-run the Phase-0.5 comparison metrics **and** produce a first **interim language clustering**,
  with an honest before/after on whether integrity work reduced the channel confound.

**Non-goals (this phase)**
- Scaling to 30–40 languages (Phase 2). Stay on the 8 seed languages.
- Finalizing the feature method — XLS-R stays *provisional* (Phase 0.5 decision); this phase asks
  "does the comparison hold up on trustworthy data?", not "which method wins."
- Full production ingest hardening (scheduling, retention) — Phase 1.
- HLS/`.m3u8` + geo-fallback capture is **optional** here (needed for German/Finnish/English/Italian
  depth); it can stay Phase 1 if the working stations yield enough independent segments.

## Workstream A — Independent 30 s segments

**Decision (user directive):** the analysis unit becomes **one ~30 s clean-speech segment per
recording**, so no two segments share a recording/speaker/channel. Segments are **independent of one
another**, which kills within-recording pseudoreplication and makes the confound check, dispersion,
and leave-one-out honest.

Implications:
- **Breadth over depth:** prioritize **many recordings across many stations and times** per language
  over many windows per clip. Target an explicit budget of *independent* 30 s segments/language
  across a minimum number of distinct stations (see open decisions).
- **≥30 s clean speech per recording required.** The collector's VAD speech gate (currently ≥5 s)
  must rise so a recording yields a full 30 s window; record longer raw clips if needed.
- **Reuse:** `clean/window.py` already cuts 30 s windows; `pipeline.py` already carries
  sub-clip→recording→station provenance. The change is *selection* (one window per recording) +
  *collection breadth*, not new windowing.

Open: strictly one-per-recording vs a few **non-adjacent** windows/recording with hierarchical
aggregation (recording → language); segment-selection rule (first valid / centered / random);
sample budget (segments/language, min stations/language).

## Workstream B — LLM segment verification (speech / music / other-language)

**Decision (user directive):** a **light LLM classifier** verifies each candidate 30 s segment is
genuinely **speech in the target language** — labelling `{speech | music | other-language | other}` —
as a data-cleaning gate. This is the **audio-based** language/content check the prior findings called
the only robust fix past the metadata ceiling, and it should also catch music/jingle leakage (the
suspected driver of the Italian/Spanish high-nPVI artefact).

Candidate approaches (pick in the brainstorm):
1. **ASR-first (recommended lean, cheap, reuses the OpenAI key in `.env`):** Whisper transcribes the
   30 s → Whisper's built-in **language ID** decides language; a light LLM (e.g. `gpt-4o-mini`,
   structured output) judges the transcript for "fluent target-language speech vs music/noise/foreign."
2. **Audio-native multimodal LLM** (e.g. an audio-capable GPT/Gemini) classifies the clip directly —
   simplest, but more cost/latency and a heavier dependency.
3. **Hybrid:** a small pretrained **speech-vs-music** audio tagger for the speech gate + Whisper
   language-ID, with the LLM only as a tie-breaker — cheapest at scale.

Design points: this **supersedes/augments** the metadata `is_in_language` guard with audio evidence;
fail policy (drop-uncertain vs keep-and-flag) is a real choice; store the per-segment verdict +
confidence in the segments table; log drops per language so coverage loss is visible. Provenance
convention (spec §9) applies — cite Whisper (Radford et al. 2022, arXiv:2212.04356) and the
classifier/model used.

Open: which approach; cost/latency budget; thresholds; how to treat low-confidence Whisper language
ID; fail-open vs fail-closed.

## Workstream C — Outlier detection (deferred robustness strand)

Per language, **after** verification, flag anomalous segments before aggregation. This is where the
**swappable robustness/Validator interface** deferred from Phase 0.5 gets built — mirror the
`FeatureExtractor` ABC (`extends src/musiclang/validation/`).

Candidate methods (swappable, evaluate ≥1):
- **Robust distance-from-centroid:** per-language median + **MAD**-scaled distance in feature space;
  flag segments beyond a threshold (e.g. >3 robust z). (MAD: Leys et al. 2013,
  https://doi.org/10.1016/j.jesp.2013.03.013.)
- **Robust covariance / Mahalanobis** (e.g. Minimum Covariance Determinant) for multivariate features.
- **Isolation Forest** (Liu et al. 2008, https://doi.org/10.1109/ICDM.2008.17) as a model-based option.

Apply in **both** spaces (prosody scalars; SSL embedding), since the right space is itself a question.
Output: per-segment outlier flag; aggregation excludes or down-weights flagged segments.

## Workstream D — Robust aggregation & validation

- **Robust aggregation:** per-language **median** (or trimmed mean) instead of mean; report
  **dispersion** (MAD/IQR) alongside central tendency.
- **Stability:** leave-one-**segment**-out and leave-one-**station**-out stability of the proximity
  matrix; bootstrap confidence intervals on the headline metrics.
- **Re-run the confound check** — now that segments are independent (one per recording), does
  **language** separation finally exceed **station**? This is the phase's central test.
- Build these as **swappable validation components** (the same robustness interface as Workstream C),
  configured per run — per the design spec §7.

## Outputs (deliverables)

1. **Re-run comparison metrics** on the cleaned/independent/outlier-filtered data: rhythm-class
   silhouette, within/between separation, and the Glottolog **Mantel** test, for **baseline + XLS-R**
   — *plus* the **station/channel confound** numbers (the key before/after).
2. **Interim language clustering** (the user-requested artefact): a **dendrogram + MDS/UMAP** of the
   8 languages — the project's first real "language proximity" picture — colored by rhythm class and
   annotated by family, for both methods.
3. **Data-integrity report** (`docs/data-integrity-findings.md`): segments surviving verification +
   outlier filtering per language; **before/after** metric + confound comparison (did integrity work
   reduce the channel confound and/or sharpen typology/family agreement?); honest coverage caveats.

## What the harness already provides (reuse, don't rebuild)

- `clean/window.py` (30 s windows), `pipeline.py` (clip→segment→features, provenance), the
  `prosody_acoustic` + `ssl_embedding` extractors, `proximity/{distance,embedding}` (centroid+cosine,
  standardize+euclidean, `linkage_matrix`, `mds_2d`), `validation/{proximity_agreement,family_tree}`
  (silhouette, within/between, **`confound_report`**, Mantel), `ingest/{radio,manifest,language_filter}`
  (capital selection, provenance manifest, metadata guard), and the collector.
- **Per-length embedding caches** exist (`data/clip_embeddings_xlsr_*.parquet`); note XLS-R-300m CPU
  inference is **slow (~78 s/full-clip)** — embed-once + cache, run as a controller-owned background
  job (see the project memory note).

## Open design decisions for the next session's brainstorm

1. Segment unit: strictly one 30 s/recording (max independence) vs a few non-adjacent windows/recording
   with hierarchical aggregation. *(lean: one/recording.)*
2. Sample budget: target independent segments/language and minimum distinct stations/language.
3. LLM verification approach: ASR+LLM vs audio-LLM vs hybrid *(lean: Whisper lang-ID + light LLM
   transcript check)*; fail-open vs fail-closed; thresholds.
4. Outlier method(s) and which feature space (prosody / embedding / both).
5. Robust aggregation: median vs trimmed mean; dispersion measure.
6. Whether to add HLS/geo capture **now** (to give German/Finnish/English/Italian enough independent
   segments) or defer to Phase 1.

## Success criteria

- **Primary:** the integrity pipeline **reduces the channel confound** (language gap > station gap, or
  a clear narrowing) and/or **improves** typology/family agreement vs the Phase-0.5 numbers — reported
  before/after with stability/CIs, not single points.
- **Secondary:** a credible **interim language clustering** with quantified stability
  (leave-one-station-out), honestly captioned.

## Relation to Phase 1 and beyond

This de-risks Phase 1's `clean`/`features` stages and bakes in the robustness controls before scaling.
After this phase, Phase 1 hardens + scales the **integrity pipeline** (independent verified segments +
outlier filtering + robust aggregation) to ~30–40 languages, and the feature-method decision
(XLS-R provisional) is revisited on trustworthy data. The swappable robustness/Validator interface
built here is the one the design spec §7 calls for.

---

*Companion documents: `docs/feature-exploration-findings.md` (Phase 0.5 decision — the motivation),
`docs/capital-stations-findings.md` (the metadata language-ID ceiling),
`docs/phase1-handoff.md` (Phase-1 kickoff decisions),
`docs/superpowers/specs/2026-06-24-music-of-languages-design.md` (design spec; §7 robustness/validation).*

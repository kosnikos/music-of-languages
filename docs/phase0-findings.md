# Phase 0 Baseline Findings & Feature-Exploration Agenda

**Status:** Phase 0 complete — baseline validated end-to-end, typology only weakly reproduced
**Date:** 2026-06-24
**Branch:** `phase0-exploration`
**Author:** Konstantinos (with Claude)

This document synthesizes Phase 0 ("Spike & method selection") of *The Music of
Languages* (see `docs/superpowers/specs/2026-06-24-music-of-languages-design.md`,
§8 roadmap). It records what the alignment-free prosody baseline produced on real
internet-radio audio for the 8 seed languages, how well it reproduces classical
rhythm typology, what confounds we saw, and — most importantly — the **prioritized
research agenda** for the Feature Exploration cycle (Phase 0.5).

The final feature-method decision is **not** made here. Phase 0's job is to deliver
a working pipeline + harness, one validated baseline, real data-sourcing experience,
and an evidence-backed agenda. The method choice is the output of the Feature
Exploration cycle (`docs/feature-exploration-cycle.md`).

All numbers below are pulled from the executed Phase 0 artifacts:
`data/lang_features.parquet` (the aggregated language table), notebooks
`01_explore_features`, `02_validate_typology`, `03_candidate_methods_feasibility`,
and the reference table in `src/musiclang/validation/typology.py`. Nothing here is
invented; every figure traces back to one of those artifacts.

---

## 1. Data coverage

We collected short clips per seed language from internet radio (talk/news biased),
ran each through loudness normalization + Silero VAD, and kept only clips with
≥5 s of detected speech. Usable-clip counts and approximate clean-speech budget
(per-language `duration_s_mean` from the parquet × usable-clip count):

| Language | Rhythm class | Usable clips | Clean-speech mean (s) | Approx. clean speech |
|---|---|---:|---:|---:|
| english | stress | 4 | 57.6 | ~230 s |
| german | stress | 2 | 59.4 | ~119 s |
| polish | stress | 4 | 45.6 | ~183 s |
| french | syllable | 4 | 57.2 | ~229 s |
| spanish | syllable | 4 | 54.3 | ~217 s |
| italian | syllable | 4 | 52.6 | ~210 s |
| greek | intermediate | 5 | 54.3 | ~271 s |
| finnish | intermediate | 1 | 55.4 | ~55 s |

**Totals:** 28 usable clips, ~1,514 s (~25 minutes) of clean speech across 8
languages. All 8 seed languages obtained *some* data — coverage is complete in
breadth but very thin in depth.

**Thin / problematic coverage and why:**

- **German (2 clips — thin).** German's top-5 talk/news stations (all WDR/BR public
  broadcasters) **failed** from this location: DNS-resolution failures and/or
  geo-blocking. German data was salvaged only via a fallback station (Kontrafunk).
  Two clips means near-zero within-language dispersion and low confidence.
- **Finnish (1 clip — very thin).** Finnish public radio (Yle) is served over
  HLS / `.m3u8`, which the simple `ffmpeg -i <url>` capture path does not handle.
  Only a single usable clip survived the VAD gate. Its within-language std is
  literally 0 (one sample → no spread), which is a coverage artifact, not a finding.
- **HLS / `.m3u8` streams failed broadly.** Yle (Finnish) and BBC streams use HLS
  playlists; some other stations returned 403/404. The Phase 0 ingest helper
  captures only plain progressive streams, so HLS endpoints were effectively
  unreachable. This is the dominant data-sourcing limitation surfaced by Phase 0.

The depth is enough to exercise the full pipeline and the validation harness end to
end; it is **not** enough to stabilize the rhythm metrics (the Arvaniti concern, §4).
Confidence is markedly lower for German and Finnish than for the 4–5-clip languages.

---

## 2. Feature results (rhythm space)

Per-language aggregated alignment-free prosody features
(`data/lang_features.parquet`, means; visualized in `notebooks/01_explore_features`
as a %V × nPVI-v scatter colored by rhythm class):

| Language | class | percent_v | npvi_v | rpvi_c | varco_c | varco_v |
|---|---|---:|---:|---:|---:|---:|
| english | stress | 36.9 | 73.2 | 0.14 | 92.4 | 76.6 |
| german | stress | 34.3 | 78.4 | 0.17 | 95.3 | 81.9 |
| polish | stress | 42.1 | 78.9 | 0.13 | 100.3 | 79.6 |
| french | syllable | 42.7 | 66.9 | 0.09 | 83.2 | 84.4 |
| spanish | syllable | 39.4 | 79.8 | 0.13 | 100.3 | 90.6 |
| italian | syllable | 35.9 | 78.3 | 0.17 | 116.1 | 91.5 |
| greek | intermediate | 42.5 | 79.8 | 0.12 | 107.5 | 86.6 |
| finnish | intermediate | 39.2 | 69.9 | 0.10 | 92.7 | 79.8 |

**The rhythm-space picture (from notebook 01):**

- **%V is compressed.** Computed `percent_v` lands in a narrow ~34–43 band. The
  classical literature spreads %V much more widely between rhythm classes; this
  compression is the signature of alignment-free, automatic vocalic/consonantal
  proportion estimation (no forced phone boundaries) rather than a property of the
  languages.
- **nPVI-v does not separate cleanly by class.** Values cluster tightly in the high
  60s–low 80s with no clean stress-vs-syllable gap. Notably, syllable-timed Italian
  (78.3) and Spanish (79.8) sit *as high as or higher than* every stress-timed
  language — the opposite of the classical expectation that stress-timed languages
  carry higher vocalic nPVI.
- **Varcos are highly variable** (varco_c spans ~83–116; varco_v ~77–92) and show no
  clean class ordering either; Italian and Greek top the varco_c range.

The features compute correctly and the pipeline produces a coherent rhythm space —
but the space does **not** visually separate the rhythm classes the way the
literature predicts.

---

## 3. Typology agreement

Validation against the reference typology
(`src/musiclang/validation/typology.py`; computed in `notebooks/02_validate_typology`):

| Metric | Value | Expected sign | Reading |
|---|---:|---|---|
| `class_separation` | **+1.79** | positive | correct direction, but **very weak** |
| `spearman_against_reference` | **−0.30** | positive (→ +1) | **negative — ranking partially inverts** |

**`class_separation` = +1.79** — mean vocalic nPVI of stress-timed languages minus
mean of syllable-timed. The sign is *correct* (stress-timed marginally higher), so
the baseline points in the right direction. But the magnitude is tiny: the published
literature shows class separations on the order of **15–30 nPVI units**. A +1.79 gap
is within noise given our dispersion and thin coverage — directionally right, not
statistically meaningful.

**`spearman_against_reference` = −0.30** — Spearman correlation between our computed
vocalic nPVI and the Grabe & Low (2002) reference values on the 5 shared languages
(english, german, polish, french, spanish). It is **negative**: the computed ranking
*partially inverts* the reference ranking. Concretely:

| Language | computed npvi_v | reference (G&L 2002) |
|---|---:|---:|
| english | 73.2 | 57.2 |
| german | 78.4 | 59.7 |
| polish | 78.9 | 46.6 |
| french | 66.9 | 43.5 |
| spanish | 79.8 | 29.7 |

- Computed order, high→low: **spanish, polish, german, english, french**
- Reference order, high→low: **german, english, polish, french, spanish**

Spanish is the worst inversion — the reference ranks it *lowest* (29.7), our pipeline
ranks it *highest* (79.8). The whole computed scale is also shifted up (~67–80 vs the
reference ~30–60), consistent with alignment-free segmentation inflating nPVI.

**Dendrogram / MDS (notebook 02).** Standardizing the 5 rhythm columns and clustering
(Ward linkage) / projecting to 2D MDS does **not** cleanly partition languages by
rhythm class. No column was dropped by `standardize()` (Finnish's zero within-language
std is stored as a valid numeric, not NaN). The clusters and the MDS scatter mix
stress-, syllable-, and intermediate-timed languages rather than grouping them.

**Where it matched / didn't.** *Matched:* the pipeline runs end to end, the harness
computes the agreement metrics, and the class-separation sign is correct. *Didn't:*
the magnitude is negligible, the rank correlation against the canonical reference is
negative, and the cluster/MDS structure does not recover the rhythm classes.

---

## 4. Confound observations (the Arvaniti risk)

The spec (§3, §7, §10) flags **metric instability** (Arvaniti 2012) — rhythm metrics
shift with speaker, speaking style, and rate, sometimes as much as they shift between
languages — as the project's biggest threat. Phase 0 shows that risk squarely,
compounded by alignment-free automatic C/V segmentation being an approximation.

- **Italian & Spanish high-nPVI artefact.** Both syllable-timed languages sit
  unexpectedly high in computed nPVI-v (78.3 / 79.8), matching or exceeding the
  stress-timed languages. This is the single most likely driver of the negative
  Spearman. The probable cause is a **radio artefact**: music/jingle leakage past the
  VAD, codec/channel effects, or automatic-segmentation error inflating vocalic-
  interval variability. It is unlikely to be a real property of Italian or Spanish.
- **%V compression.** Computed `percent_v` is squeezed into ~34–43 (§2), so the most
  robust classical metric carries less discriminative signal here than it should —
  expected from alignment-free estimation, but it weakens the baseline.
- **Thin coverage → unreliable dispersion.** German (2 clips) and Finnish (1 clip)
  yield near-zero or meaningless within-language spread (Finnish `duration_s_std` =
  0.00; German = 0.30), so their means are low-confidence point estimates. Languages
  with 4–5 clips show real dispersion (e.g. Polish `duration_s_std` = 12.24, Italian
  = 6.24), which is itself evidence of clip-to-clip / station / speaker / rate
  variation — exactly the wobble the spec's "report dispersion alongside central
  tendency" defense exists to surface. With only 1–5 clips per language, that wobble
  is large relative to any between-language signal.

Net: the between-language signal we want is currently swamped by within-language and
channel noise — the textbook Arvaniti failure mode, made worse by thin data and
alignment-free segmentation.

---

## 5. Candidate-method assessment

From `notebooks/03_candidate_methods_feasibility` (Task 16). Phase 0 implemented the
alignment-free baseline and *assessed* heavier methods behind the same
`FeatureExtractor` interface.

| Method | Interpretability | Infra cost | Expected fidelity | Status in Phase 0 |
|---|---|---|---|---|
| Alignment-free baseline (%V / nPVI / Varco) | High | Minimal | Weak (this phase's result) | Implemented + validated |
| SSL embeddings (wav2vec2 / XLS-R / ECAPA) | Low–Medium | Low (inference-only, CPU) | Medium–High | **Probe executed** |
| MFA rhythm metrics | High | High (conda/Kaldi + ASR) | High in principle | Documented assessment |
| Envelope-RNN (Deloche) | Medium | Medium (training, 50+ clips/lang) | Medium–High | Documented assessment |

- **SSL embeddings — executed probe, recommended first.** `facebook/wav2vec2-base`
  was run inference-only over real clips: a 768-d mean-pooled embedding per clip,
  cosine distance works (english ↔ finnish ≈ **0.12**, matching the controller smoke
  test 0.1193). No training, no labels, CPU-friendly. Multilingual upgrades (XLS-R,
  VoxLingua107 ECAPA) are available off-the-shelf and plug straight into
  `proximity.distance`. This is the lowest-effort, highest-upside next step.
- **MFA — high infra weight, deferred.** Pretrained acoustic models + dictionaries
  exist for most seed languages (Greek/Finnish variable), but MFA needs a **verbatim
  text transcript** — our radio clips are unlabeled, so it requires an ASR stage
  first, plus a heavy conda/Kaldi/Windows install incompatible with the pip venv. The
  marginal gain over alignment-free %V/nPVI is questionable for unlabeled radio.
  Recommended only for a small read-speech calibration sub-corpus, if at all.
- **Envelope-RNN — needs training data, deferred.** Deloche et al. (2024) is the
  closest prior art, but there is no public pretrained checkpoint, and it needs
  ~50–200 clips/language (we have ~4). Revisit after the corpus is expanded.

---

## 6. Baseline verdict & exploration agenda

**Verdict.** The alignment-free prosody baseline **works end to end and validates the
pipeline + harness** — catalog → ingest → clean (VAD) → features → proximity →
validate all run on real internet-radio audio, and the agreement metrics compute. But
it only **weakly and inconsistently reproduces classical rhythm typology**:

- `class_separation` is **+1.79** — correct sign, but ~10× too small to be meaningful
  (literature: 15–30 units).
- `spearman_against_reference` is **−0.30** — the computed nPVI ranking *partially
  inverts* the Grabe & Low (2002) reference.
- Dendrogram / MDS do not recover the rhythm classes.

This is consistent with the spec's stated risk (Arvaniti instability + alignment-free
approximation) and the thin Phase 0 corpus (1–5 clips/language, German/Finnish
especially thin). The baseline is a **valid harness and a weak signal** — exactly the
"one validated baseline + research agenda" outcome Phase 0 was scoped to produce. It
is good enough to evaluate *other* methods against; it is not yet good enough to ship
a proximity space on.

**Prioritized agenda for the Feature Exploration cycle (Phase 0.5).** Implement and
compare, in this order, all behind the `FeatureExtractor` interface and all validated
against typology + family structure with the existing harness:

1. **Pretrained SSL embeddings — FIRST.** Strongest evidence and lowest cost. The
   Phase 0 probe already runs wav2vec2-base inference-only and produces sensible
   cosine distances (≈0.12 english↔finnish) with no training and no labels. Upgrade
   to a **multilingual** model (XLS-R or VoxLingua107 ECAPA), embed all clips,
   re-run `class_separation` / `spearman_against_reference` / dendrogram / MDS, and
   compare head-to-head against the alignment-free baseline. This is the first method
   the cycle should land.
2. **Expand the corpus** (prerequisite for #3, also helps #1's dispersion). Target
   50+ clips/language across multiple stations and times; this both stabilizes the
   baseline metrics and unlocks envelope-RNN training.
3. **Envelope-RNN (Deloche-style) — SECOND**, after data expansion. Closest prior art
   for "prosody alone identifies language," but needs training data we don't yet have.
4. **MFA rhythm metrics — OPTIONAL / last.** Only as a read-speech calibration check
   (labeled sub-corpus), given its ASR + conda/Kaldi infra weight on unlabeled radio.

The **final method choice is the cycle's output, not this document's.** This agenda
sets the order of investigation and the evidence behind it; the cycle decides which
method(s) — one, several, or a fusion — actually carry forward.

---

## 7. Recommended Phase 1 adjustments

Things the Phase 0 data taught us, to fix before / during the next data pass:

- **More clips per language — especially German and Finnish.** 1–5 clips is too thin
  to stabilize rhythm metrics or report meaningful dispersion. Aim for tens of clips
  per language so the Arvaniti wobble averages out.
- **More stations per language** so the pipeline survives individual-station failure.
  German lost its top-5 broadcasters to DNS/geo issues and was rescued by one
  fallback; a deeper station list per language prevents single points of failure.
- **Robust stream handling.** Add **HLS / `.m3u8`** capture (e.g. `streamlink` or an
  `ffmpeg` HLS path) — this alone would have recovered Finnish (Yle) and BBC. Add
  **geo-fallback / mirror resolution**, retries, and graceful handling of 403/404 so
  transient and location-specific failures don't drop a language to 1–2 clips.
- **Stronger music/speech filtering** to fix the Italian/Spanish high-nPVI artefact.
  Tighten the VAD gate and add an explicit music-vs-speech classifier so jingles and
  music beds don't leak into the vocalic-interval statistics.
- **Multi-station, multi-time sampling for genuine dispersion.** Sample each language
  across several stations and times of day so the reported `*_std` reflects real
  speaker/style/rate variation, supporting the spec's "aggregate over variation"
  robustness defense rather than coverage noise.
- **Metric notes for the feature cycle.** %V is compressed and nPVI is inflated/
  inverted under alignment-free segmentation here. Keep the metrics in the comparison
  harness as a baseline, but do not treat the current alignment-free nPVI ranking as
  trustworthy until SSL (and/or aligned) methods corroborate it.

---

*Companion documents: `docs/phase1-handoff.md` (kickoff decisions & rationale),
`docs/feature-exploration-cycle.md` (Phase 0.5 brief, where the method decision is
made), `docs/superpowers/specs/2026-06-24-music-of-languages-design.md` (design spec).*

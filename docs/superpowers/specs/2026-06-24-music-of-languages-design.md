# The Music of Languages — Design Spec

**Status:** Approved design (pre-implementation)
**Date:** 2026-06-24
**Author:** Konstantinos (with Claude)

## 1. Summary

A data project that analyzes the **"music" (prosody) of spoken languages**, uses ML to
measure how *similar languages sound*, and visualizes the result as **colored maps** and
**proximity diagrams** (dendrograms, MDS/UMAP). The end deliverable is a **published blog
report**.

- **Primary feature axis:** prosody & melody — rhythm, intonation, pitch, tempo.
- **Modular by design:** prosody is the first "vertical"; phonetic-inventory and
  self-supervised-embedding verticals can be added later and **combined**.
- **v1 granularity:** language level (~30–40 European languages), architected to drill
  down to **sub-country regions / dialects** later.
- **Inputs:** spoken-language audio sampled from **internet radio**.
- **Stack:** Python.
- **Method selection is evidence-driven:** an exploration phase (Jupyter + literature)
  decides which feature-extraction method(s) we actually use.

## 2. Goals & non-goals

**Goals**
- Produce a defensible, reproducible language-proximity space derived from how languages *sound*.
- Tell a compelling, honest story in a blog (maps + proximities + "does sound match family?").
- Build a modular, extensible pipeline (more verticals, finer granularity later).

**Non-goals (v1)**
- Sub-country/regional/dialect granularity (designed for, not delivered in v1).
- Phonetic-inventory and SSL-embedding verticals (v2).
- Real-time / streaming analysis. Everything is batch over collected samples.
- A polished product/UI. Outputs are figures + a written report.

## 3. Background — what the research established

(From the deep-research pass, 2026-06-24; primary sources noted.)

**The canonical method — duration-based rhythm typology:**
- Ramus, Nespor & Mehler (1999): segment speech into vocalic/consonantal intervals; compute
  **%V, ΔC, ΔV**. On the %V-vs-ΔC plane, languages separate into stress-/syllable-/mora-timed.
- Grabe & Low (2002): the **PVI** family — vocalic **nPVI** (rate-normalized), intervocalic
  **rPVI**. A 2D nPVI×rPVI space separates rhythm types better than any single index.
- Dellwo (2006): **VarcoC/VarcoV** = rate-normalized deltas (coefficient-of-variation style).

**Closest prior art to this project:**
- Deloche, Bonnasse-Gahot & Gervain (2024, arXiv:2401.14416): an RNN trained on *only*
  amplitude envelopes + voicing identified 21 languages from 10s clips (~40% top-1 / ~66%
  top-3) and produced learned "rhythm maps" consistent with classical typology but richer
  than a binary split. Validates the premise that prosody alone carries language identity.

**Three findings that shape the design:**
1. **%V is the most robust metric; raw ΔC/ΔV are fragile** under automatic segmentation and
   are confounded by **speech rate** → prefer rate-normalized metrics (Varcos).
2. **Rhythm is a continuum, not discrete classes** → model proximity as a continuous space;
   use multiple metrics as axes.
3. **⚠️ Metrics are unstable (Arvaniti 2012):** they shift with speaking style, speaker, and
   rate — sometimes as much as they shift between languages. **This is the project's biggest
   threat** and motivates the robustness/validation work in §7.

**Tooling surfaced (Python-friendly):** Parselmouth (`praat-parselmouth`), DisVoice
(103 prosody features, alignment-free), De Jong & Wempe Praat script (automatic speech rate),
Correlatore (full metric battery, needs hand-segmented TextGrids). Leads for later phases
(found, not yet verified): `radio-browser.info` API + `pyradios`; Silero VAD / pyannote;
VoxLingua107 ECAPA + XLS-R.

## 4. Architecture

A **staged, modular pipeline**. Each stage reads/writes versioned artifacts on disk, so
stages are independently runnable, cacheable, and notebook-friendly ("re-run only the last
stage"). The heart is a pluggable **`FeatureExtractor` interface**.

```
catalog → ingest → clean → features → proximity → validate → viz → report
                              ▲
              pluggable FeatureExtractor implementations
```

### 4.1 Stages

| Stage | Job | Candidate tools |
|---|---|---|
| **catalog** | Select stations per language (later: per region); tag language/region | `radio-browser.info` API, `pyradios` |
| **ingest** | Record many short clips (~30–60s) per station across times → capture multiple speakers | `ffmpeg` / `streamlink` |
| **clean** | Keep speech only: VAD, music-vs-speech filter, optional diarization, loudness-normalize/resample | Silero VAD, librosa |
| **features** | Per-clip → feature vector; aggregate per language (mean + dispersion) | *pluggable (§4.2)* |
| **proximity** | Standardize features → language×language distance matrix → hierarchical clustering + MDS/UMAP | scikit-learn, scipy, umap-learn |
| **validate** | Assess robustness; compare proximity vs reference structures | *pluggable (§7)* |
| **viz** | Dendrogram, MDS/UMAP scatter, colored map (region color = 2D proximity embedding) | geopandas, plotly/folium |
| **report** | Assemble blog figures + narrative | Jupyter / Quarto |

### 4.2 FeatureExtractor interface (the modular core)

One interface, multiple implementations, selectable/combinable via config:

- **`ProsodyAcoustic`** — DisVoice + De Jong-Wempe + Parselmouth (F0/intonation, energy,
  tempo, pause/duration, %V-style vocalic proportion). **Alignment-free.** v1 primary candidate.
- **`RhythmMetricsMFA`** — forced-alignment canonical battery (%V, ΔC, ΔV, Varcos, nPVI/rPVI)
  via Montreal Forced Aligner + Correlatore-style computation. Exploration/comparison.
- **`EnvelopeEmbedding`** — Deloche-style net on amplitude envelopes + voicing. v2 vertical.
- **`SSLEmbedding`** — XLS-R / VoxLingua107 ECAPA. v2 vertical.
- **`PhoneticInventory`** — phoneme-inventory-based similarity. v2 vertical.

Each implementation: `extract(clip) -> feature_vector`, plus a per-language aggregation that
returns central tendency **and dispersion** (so we can *see* the Arvaniti wobble, not hide it).

**Method selection is deferred to Phase 0** (notebooks + literature): implement the first
candidates, check which reproduce known typology on the seed set, then decide which method(s)
to carry forward (possibly more than one, possibly combined).

## 5. Data flow & artifacts

```
stations.parquet
  → raw clips (retention-policy controlled) + clips_manifest.parquet
  → clean_segments/ + segment metadata (speaker count, durations, speech/music flags)
  → features.parquet (per-clip)
  → lang_features.parquet (aggregated per language: mean + dispersion)
  → distance_matrix.npy + embedding_2d.parquet
  → figures / report
```

Artifacts are cached per stage and keyed by config, so re-running is incremental.

## 6. Storage & retention (configurable)

Storage goes through an abstraction with a **retention policy**:

- **`permanent`** (default for now): keep recorded audio + all artifacts locally. Convenient
  for iterating and re-running feature methods without re-downloading.
- **`ephemeral`**: extract features then discard raw audio, keeping only derived features
  (+ optionally a few short illustrative clips).

The same pipeline code supports both; switching is a config change. **Default while
developing: `permanent`**, with an eye on laptop disk usage (short clips, modest hours per
language keep this small). **Legal/ToS/copyright is explicitly deferred to pre-publication** —
revisit retention + redistribution before the blog goes live. *(Not legal advice.)*

## 7. Robustness & validation (open research area)

The Arvaniti instability (§3) means **naive similarity can be spurious**. Validation is
treated as a **pluggable, configurable research area** — we document candidate techniques but
do **not** hard-wire any single approach; the choice is refined during exploration.

**Candidate techniques to evaluate (non-exhaustive, extensible):**
1. **Aggregate over variation** — many clips, many speakers, multiple stations & times per
   language; report dispersion alongside central tendency.
2. **Validate against reference structures** — correlate the audio-derived proximity matrix
   against known references: language-family trees, classical rhythm-class labels, and/or
   lexical distance (e.g. ASJP) via correlation / Mantel test. Doubles as the blog's narrative
   hook ("does how a language sounds match its family?").
3. **Outlier handling** — exclude or down-weight anomalous clips/speakers; robust aggregation.
4. **Others TBD** — to be surfaced by further research/exploration.

Design requirement: validation/robustness steps are swappable components with a common
interface, configured per run — not assumptions baked into the pipeline.

## 8. Roadmap (exploration-first)

- **Phase 0 — Spike & method selection** *(notebooks + literature review)*
  Scaffold repo; collect a small seed set spanning known rhythm classes; implement the first
  candidate feature methods in notebooks; check which reproduce known typology.
  **Gate: decide which feature method(s) and validation approach(es) to carry forward.**
  **Seed languages (agreed):** English, German, Polish, French, Spanish, Italian, Greek, Finnish.

- **Phase 1 — Thin end-to-end slice**
  Harden the chosen extractor into the pipeline; run catalog→viz on the seed languages from
  *real radio*; produce a first dendrogram + colored map. Proves the whole chain + data
  sourcing + cleaning + chosen robustness controls.

- **Phase 2 — Scale to ~30–40 European languages**
  Expand catalog, scale ingest, tune robustness controls → the v1 blog deliverable
  (maps + proximities + validation).

- **Phase 3 — Extensibility demos**
  Add a second vertical (e.g. SSL embedding) and compare/combine with prosody; regional
  drill-down in 1–2 countries as a v2 teaser.

- **Phase 4 — Blog write-up**
  Narrative + polished figures.

## 9. Tech stack & conventions

- **Language:** Python 3.11+.
- **Audio/features:** librosa, praat-parselmouth, DisVoice, silero-vad; (MFA where needed).
- **ML/analysis:** numpy, pandas, scikit-learn (clustering, MDS, distances), umap-learn,
  scipy (dendrogram, Mantel).
- **Data sourcing:** radio-browser.info via `pyradios`; `ffmpeg`/`streamlink` for capture.
- **Viz/report:** geopandas, plotly/folium; Jupyter (+ Quarto for the report).
- **Artifacts:** parquet for tabular data/manifests; per-stage cached directories keyed by config.
- **Config-driven:** which languages, which feature extractor(s), which validation steps,
  retention policy — all configuration, not code edits.

## 10. Key risks

| Risk | Mitigation |
|---|---|
| **Rhythm-metric instability** (Arvaniti) | Aggregate over many speakers/clips/times; report dispersion; validate vs reference structures (§7) |
| **Radio confounds** (music, ads, jingles, single speaker, channel/codec) | VAD + music/speech filter + diarization in `clean`; multi-station, multi-time sampling |
| **Method picks the wrong signal** (rate/content rather than "music") | Rate-normalize; method-selection gate in Phase 0 validated against known typology |
| **Storage growth on laptop** | Short clips, modest hours/language; `ephemeral` retention available |
| **Legal/ToS for publication** | Deferred to pre-publish; retention abstraction supports switching to features-only |

## 11. Open questions (to resolve during exploration)

- Which feature method(s) win Phase 0 — one, several, or a combination?
- Which validation/robustness techniques prove most reliable?
- Exact per-language sample budget (stations × clips × hours) to stabilize metrics.
- Reference structures for validation (which family-tree / typology / lexical datasets).
- v2 regional data sourcing (region-tagged stations) and dialect-vs-language confounds.

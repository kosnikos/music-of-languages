# Feature Exploration Cycle — SSL Embeddings vs. Prosody Baseline (Design Spec)

**Status:** Approved design (pre-planning)
**Date:** 2026-06-25
**Branch:** `feature-exploration-ssl`
**Author:** Konstantinos (with Claude)
**Parent docs:** `docs/superpowers/specs/2026-06-24-music-of-languages-design.md` (design §4.2, §7),
`docs/feature-exploration-cycle.md` (the Phase 0.5 brief), `docs/phase0-findings.md` (baseline result).

## 1. The question this cycle answers

Does a **multilingual self-supervised (SSL) speech embedding** — headline model
**XLS-R-300m** — reproduce rhythm typology and language-family structure **better than the
alignment-free prosody baseline**, on lightly-expanded internet-radio data, measured by
**method-agnostic proximity metrics**, and does it survive a **channel/station confound check**?

The Phase 0 baseline reproduces typology only weakly: `class_separation = +1.79` (literature:
15–30 nPVI units) and `spearman_against_reference = −0.30` (rank order partially **inverts**
Grabe & Low 2002). This cycle implements the first real challenger behind the existing
`FeatureExtractor` interface and runs an honest head-to-head.

**Output:** a decision doc (`docs/feature-exploration-findings.md`) recording the verdict and
what carries into Phase 1. This cycle does **not** finalize the whole method roster — it lands
and evaluates the SSL method (and the comparison harness) per the cycle brief's priority order.

## 2. Scope

**In scope (this cycle):**
1. A configurable HF-SSL `FeatureExtractor` (wav2vec2-base / XLS-R / HuBERT via config), headline
   run XLS-R-300m, with a small **layer sweep**.
2. An **embedding-appropriate proximity path** (per-language centroid → cosine), distinct from the
   scalar `mean+std → z-score → Euclidean` path.
3. **Method-agnostic comparison metrics** (proximity-space typology) + a **family-tree Mantel test**.
4. **Analysis-window length** as a configurable parameter, with a **mini-sweep** over L ∈ {10s, 30s, full}.
5. **Hierarchical provenance & aggregation** (sub-clip → recording → station → language) so dispersion
   is not pseudoreplicated.
6. **Light corpus expansion** via the existing collector (more clips, more stations/lang, station
   provenance recorded) — no new HLS infrastructure.
7. A **channel/station confound check**.
8. A method-comparison notebook + the decision doc.

**Out of scope (explicitly deferred):**
- The full **swappable Validator ABC** and the rest of the robustness/outlier strand: outlier
  filtering (MAD / distance-from-centroid), robust aggregation (median), leave-one-clip-out
  stability, and the *full* clip-length sensitivity suite. (This is the next cycle — the strand
  sequenced second. The metrics built here are written as plain functions, ABC-ready, not the ABC.)
- VoxLingua107 **ECAPA** language-ID embedding (the purpose-built "upper bound" — a fast follow, not now).
- **Envelope-RNN** and **MFA** feature methods.
- **Fuller data expansion** (HLS/`.m3u8` capture, geo-fallback, 50+ clips/lang) — Phase 1.

## 3. Components

All new feature/validation code follows the **provenance convention** (spec §9): every
feature-/metric-computing function cites, in its docstring, a link to the authoritative source for
its maths/reasoning (scientific preferred; Wikipedia fallback).

### 3.1 `SSLEmbeddingExtractor` (`src/musiclang/features/ssl_embedding.py`)

Implements the existing ABC (`name` property; `extract(signal, sr) -> dict[str, float]`).

- **Config:** `model_id: str = "facebook/wav2vec2-xls-r-300m"`, `layer: int` (index into
  `hidden_states`; negative allowed), `pooling: str = "mean"` (`"mean"` | `"mean_std"`),
  `device: str = "cpu"`.
- **Behavior:** lazily load the HF `AutoModel` + `AutoFeatureExtractor` (cached per `model_id`);
  resample the signal to the model's 16 kHz (already `TARGET_SAMPLE_RATE`); run with
  `output_hidden_states=True`; select `hidden_states[layer]`; pool over time
  (`mean`, or `mean⊕std` concatenation — temporal std is itself a prosodic signal); return the
  pooled vector as `{"emb_000": ..., "emb_NNN": ...}` (zero-padded indices, stable ordering).
- **`name`:** e.g. `f"ssl_{short_model}_l{layer}_{pooling}"` so outputs are namespaced and the
  cache key is unambiguous.
- **Normalization:** `extract` returns the **raw** pooled vector. L2-normalization happens at
  centroid aggregation (§3.3), so the per-clip feature stays faithful.
- **Provenance citations:** wav2vec 2.0 (Baevski et al. 2020, arXiv:2006.11477); XLS-R
  (Babu et al. 2021, arXiv:2111.09296); HuBERT (Hsu et al. 2021, arXiv:2106.07447); the
  mean/mean-std pooling + cosine-distance choice.

**Layer sweep.** XLS-R-300m exposes 25 hidden states (embedding + 24 layers); mid layers typically
carry the most phonetic/linguistic information, the last layer is more pretraining-objective-specific.
We sweep a small set (default `{8, 12, 16, 20, last}`, configurable) and **report which layer
maximizes typology separation** rather than assuming one. `wav2vec2-base` (12 layers) is kept as the
already-probed fast anchor.

### 3.2 Windowing & provenance (`src/musiclang/clean/window.py` + manifest)

- **`window_signal(signal, sr, length_s, hop_s=None, min_s=None) -> list[Window]`** — split a
  cleaned/concatenated speech signal into windows; **non-overlapping by default** (`hop_s = length_s`;
  overlap would only deepen sample correlation). A trailing window shorter than `min_s` is dropped.
  Each `Window` carries `(start_s, samples)`.
- **Provenance.** Every analysis unit is traceable: `language → station → parent_recording (clip_id)
  → window_index → start_s → length_s`. Two artifacts:
  - **`data/clips_manifest.parquet`** — one row per recorded clip:
    `clip_id, language, station_name, station_url, country, recorded_at, duration_s, path`.
  - **`data/segments.parquet`** — one row per analysis window:
    `segment_id, clip_id, language, station_name, window_index, start_s, length_s`.
- `full` length is represented as a single window spanning the whole cleaned clip (so the same code
  path serves all sweep values).

### 3.3 Embedding proximity path (`src/musiclang/proximity/`)

Embeddings must **not** ride the scalar `aggregate (mean+std) → standardize (z-score, drop-NaN) →
Euclidean` path (per-dim std and per-dim z-scoring of an embedding is noise).

- **`language_centroids(emb_df, group="language", recording_col="clip_id") -> DataFrame`** — per
  language, the mean of **L2-normalized** per-segment embeddings. Default weighting averages **per
  recording first, then across recordings** (so a chatty station can't dominate a language's
  centroid); a `flat` option (mean over all segments) is available for comparison.
- **Distance:** the existing `distance_matrix(centroids, metric="cosine")` (scipy `pdist` already
  supports cosine).
- **Caching:** per-segment embeddings are written to
  `data/clip_embeddings_<model>_<layer>_<pool>.parquet` keyed by `segment_id`. XLS-R inference is the
  expensive step — computed once, reused across the layer/length analyses where possible.

The baseline keeps its existing scalar path unchanged; only the *comparison metric* (§3.4) is shared.

### 3.4 Validation additions (functions now, ABC-ready)

The existing scalar `class_separation` / `spearman_against_reference` (`validation/typology.py`)
require a single scalar nPVI per language, so they **cannot** evaluate an embedding. They remain as
**baseline-only diagnostics**. The head-to-head uses metrics that operate on a **distance matrix +
class labels** and therefore apply to *any* method:

- **`validation/proximity_agreement.py`:**
  - `class_silhouette(dist_df, labels) -> float` — silhouette of the rhythm-class labelling using the
    precomputed distance matrix (sklearn `silhouette_score(metric="precomputed")`). Higher = classes
    better separated. (Rousseeuw 1987.)
  - `within_between_separation(dist_df, labels) -> dict` — mean within-class vs mean between-class
    distance and their gap/ratio; sign and magnitude are interpretable across methods.
  - Dendrogram/MDS-colored-by-class are produced in the comparison notebook from existing
    `linkage_matrix` / `mds_2d`.
- **`validation/family_tree.py`:**
  - `reference_distance_matrix(languages) -> DataFrame` — an 8×8 **genealogical** distance matrix
    hand-curated from the **Glottolog** classification (Germanic: english, german; Slavic: polish;
    Romance: french, spanish, italian; Hellenic: greek; Uralic: finnish), distances derived from
    shared-classification depth. Transparent and fully documented in the docstring; **ASJP lexical
    distances** noted as a richer optional alternative. (Glottolog — Hammarström et al.)
  - `mantel_test(dist_a, dist_b, method="pearson", permutations=10000, seed=0) -> (r, p)` — a small
    self-contained permutation Mantel test on the off-diagonal entries (avoids a `skbio` dependency).
    (Mantel 1967.)

### 3.5 Light corpus expansion (`scripts/collect_sample.py` + `ingest/`)

Re-run the existing collector for **more clips and more stations per language** (reusing
`find_capital_stations` / `find_stations`, the OpenAI language guard, VAD cleaning), targeting
**~10–15 usable clips/lang across ≥2–3 distinct stations**. The collector now **writes the
`clips_manifest.parquet` with station provenance** (required for the confound check) in addition to
the per-clip WAVs. No HLS/`.m3u8` capture, geo-fallback, or scheduling — those stay Phase 1.

### 3.6 Channel/station confound check

Using per-segment embeddings with **station** labels: does same-language-different-station audio
cluster together more than different-language-similar-channel audio? Reported as (a) silhouette by
**language** vs silhouette by **station**, and (b) within-language-between-station vs
between-language distances. If station/channel dominates language, the SSL method is measuring the
channel, not the language's sound — a documented, decision-relevant outcome.

### 3.7 Length mini-sweep (the "what length is appropriate" experiment)

For **L ∈ {10s, 30s, full}**: window → extract (both methods) → aggregate → proximity → metrics.
Report, per method and per L: **typology separation** (silhouette, within/between, Mantel r) and
**within-recording stability** (spread of per-window features within a single recording — a
robustness signal that isolates short-sample noise from speaker/station variation). This answers
"what window length is right for our task" empirically and directly informs the Phase-1 ingest
decision (how long to record).

## 4. Evaluation protocol & success criteria

For each method (baseline, SSL across the layer/length sweep) on the **same** lightly-expanded corpus:

1. Build the per-language proximity matrix (baseline: standardized rhythm features → Euclidean;
   SSL: L2-normalized centroid → cosine).
2. **Typology agreement** — `class_silhouette`, `within_between_separation`; dendrogram + MDS colored
   by rhythm class. (Baseline also reports its legacy scalar `class_separation` /
   `spearman_against_reference` for continuity.)
3. **Family-tree agreement** — `mantel_test` of the proximity matrix vs the Glottolog reference.
4. **Robustness** — within-recording stability and between-recording/station dispersion (hierarchical,
   not pseudoreplicated), reported alongside central tendency.
5. **Confound resistance** — §3.6.
6. **Interpretability** — noted honestly (the SSL embedding is less interpretable than %V/nPVI).

**SSL "wins"** if its proximity matrix separates rhythm classes better than baseline (higher
silhouette; within < between) **and/or** correlates more with the family tree (higher Mantel r),
**while** language clustering dominates station clustering in the confound check. A tie, or "SSL
separates languages but largely via channel/phonetics rather than prosody," is a legitimate,
documented verdict — not a failure.

## 5. Data flow & artifacts

```
clips_manifest.parquet        (per clip: language, station, time, path)
  → segments.parquet          (per window: clip_id, station, window_index, start, length)
  → clip_embeddings_<model>_<layer>_<pool>.parquet   (cached per-segment SSL vectors)
  → per-language centroid tables  +  lang_features.parquet (baseline, existing)
  → distance matrices per method  → silhouette / within-between / Mantel
  → comparison figures (dendrogram, MDS) + length-sweep table
  → docs/feature-exploration-findings.md (the decision doc)
```

## 6. Testing strategy (TDD)

- **`SSLEmbeddingExtractor`** — monkeypatch the HF model/feature-extractor to return deterministic
  `hidden_states`; assert interface compliance, output shape, correct layer selection, `mean` vs
  `mean_std` pooling, stable key ordering, and `name`. One **real-model integration test** marked
  slow/feasibility exercises XLS-R end-to-end on a short synthetic signal.
- **`window_signal`** — pure-function tests: window counts, lengths, non-overlap, short-tail drop,
  `full` single-window case, provenance indices.
- **`proximity_agreement`** — synthetic distance matrices with known class structure: silhouette sign,
  within < between when classes are separated, and the degenerate single-class case.
- **`mantel_test`** — identical matrices → r≈1, small p; shuffled → r≈0; known small example.
- **`reference_distance_matrix`** — sanity: `d(spanish, italian) < d(spanish, finnish)`; symmetry;
  zero diagonal.
- **`language_centroids`** — known vectors: L2-normalization, per-recording-then-per-language
  weighting vs flat.

## 7. Dependencies

- Graduate `transformers` from the `feasibility` optional extra to a **core dependency** (XLS-R +
  wav2vec2 loading). `torch`/`torchaudio` already pinned (`>=2.2,<2.12`, CPU wheels).
- `silhouette_score` from `scikit-learn` (already present). Mantel is self-implemented (no `skbio`).
- No `speechbrain` (ECAPA deferred). No new audio deps (windowing is numpy slicing on already-loaded
  signals).

## 8. Risks & open questions

- **XLS-R-300m CPU inference time** over a few hundred segments × the layer sweep. Mitigation:
  cache per-segment embeddings (§3.3); the layer sweep reuses one forward pass via
  `output_hidden_states=True`; `wav2vec2-base` is the fast anchor if 300m is impractical.
- **Centroid weighting** (per-recording vs flat) can shift results when station clip-counts are
  uneven; default is per-recording, with the flat variant reported for sensitivity.
- **Glottolog hand-curated distances are coarse** (depth-based, 8 languages); acknowledged in the
  doc, with ASJP as the richer alternative if the Mantel signal warrants it.
- **Existing 31 clips may lack station provenance**; the light expansion re-collects with the
  manifest, superseding/augmenting the old clips so the confound check has the metadata it needs.
- **Window length fairness** — 10s windows may handicap the rhythm baseline more than SSL; this is
  exactly what the length sweep surfaces, and we report per-method rather than forcing one length.

## 9. Deliverables

- `src/musiclang/features/ssl_embedding.py` (the SSL extractor).
- `src/musiclang/clean/window.py` (windowing) + manifest/segment provenance.
- `src/musiclang/proximity/` additions (centroid aggregation; cosine path wiring).
- `src/musiclang/validation/proximity_agreement.py` and `validation/family_tree.py`.
- Collector changes for light expansion + provenance.
- `notebooks/04_ssl_vs_prosody.ipynb` (the method comparison + length sweep + confound check).
- `docs/feature-exploration-findings.md` — the decision doc (the cycle's output).

## 10. Execution

Spec → implementation plan (writing-plans) → **subagent-driven development** against the plan, TDD
throughout, with review checkpoints.

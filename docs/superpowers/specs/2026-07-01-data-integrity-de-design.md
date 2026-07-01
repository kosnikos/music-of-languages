# Data Integrity Phase — Workstreams D + E — Design Spec

**Date:** 2026-07-01
**Branch:** `data-integrity-brief` (HEAD 77293be at spec time)
**Package:** `musiclang` (src-layout, `src/musiclang/`)
**Phase brief:** `docs/data-integrity-phase.md` (Workstreams D + E)
**Motivation:** `docs/feature-exploration-findings.md` §4–§5 (the method comparison + the station/language confound)
**Predecessor cycles:** Workstream A (`docs/source-evaluation.md`), Workstreams B+C (verified-segment dataset)

---

## 1. Purpose

Phase 0.5 proved the comparison harness works but is **data- and confound-limited**: on the 55-clip
corpus the SSL "win" over the prosody baseline was weak, and segment-level geometry clustered **more
by recording station than by language** (within/between gap **0.049** station vs **0.012** language;
`feature-exploration-findings.md` §5). Workstreams A–C fixed the *data*: we now have **178 verified,
independent, clean 30 s speech segments across 59 channels** (one segment per recording → no
within-recording pseudoreplication). D + E build the **outlier-detection + robust-aggregation** strand
deferred from Phase 0.5 and then **re-measure**, answering the phase's central question:

> With independent verified segments, outlier filtering, and robust aggregation, does **language**
> separation now beat **station** separation — and does typology/family agreement sharpen — versus the
> Phase-0.5 numbers?

**This phase does not finalize the feature method.** XLS-R stays *provisional* (Phase-0.5 decision). We
ask "does the comparison hold up on trustworthy data?", not "which method wins."

### Scope decisions (locked in the brainstorm, 2026-07-01)

1. **One combined D+E spec/plan**, executed subagent-driven. The XLS-R embedding job launches early as a
   controller background job so it finishes while the outlier/robustness code is built and tested.
2. **Compute both feature methods** — prosody baseline (`ProsodyAcousticExtractor`, 16 scalars) and
   XLS-R embeddings (`SSLEmbeddingExtractor`), embed-once-and-cache.
3. **Outlier handling = report-both.** Flag outliers, then compute every headline result **with** and
   **without** exclusion and report the delta. No silent filtering (protects thin languages; makes the
   outlier strand's effect measurable).
4. **Feature config = fixed headline, cheap sensitivity.** Headline = the Phase-0.5 provisional
   (XLS-R-300m, layer 16, 30 s, mean-pool; prosody 16 scalars). Cache layers **{12, 16, last}** in the
   **same forward pass** (nearly free) so a layer-sensitivity check is available without extra compute.
5. **Swappable `Validator`/`OutlierDetector` ABC** mirroring `FeatureExtractor` (design spec §7).
6. **Optional components included:** IsolationForest as a swappable 2nd detector (prosody space);
   per-channel centroid weighting (confound mitigation). **Deferred:** UMAP segment viz (MDS covers it);
   Mahalanobis/MCD (infeasible at 1024-dim, borderline at 16-dim/n=14).

---

## 2. Input data

`data/segments/segments_manifest_final.parquet` — 178 verified segments, 59 distinct channels, 0
duplicate recordings, all `tagger_music <= 0.071` (clean). **Gitignored.** Wavs live at
`data/segments/<lang>/` and `data/segments_topup/<lang>/`; absolute paths are in the `path` column.

**Columns:** `segment_id, language, source, channel_id, recording_ref, recorded_at, clean_speech_s,
path, label, confidence, detected_language, transcript, tagger_speech, tagger_music, stage_decided`.

**Per-language coverage:** english/german/french/italian/finnish 25, polish 24, greek 15, spanish 14.
Greek and Spanish are **carpet-scarce** (documented caveat — music-bed broadcast talk is common there).

**Two consequences of "one segment per recording":**
- SSL `language_centroids(weighting="recording")` ≡ `weighting="flat"` here (each recording has exactly
  one segment). The *new* lever is `weighting="channel"` (below), since channels carry ~3 segments each.
- With ~3 segments/channel (vs Phase-0.5's 42/45 singleton stations) the **station confound check is now
  a real measured effect**, not near-degenerate.

The wavs are **already cleaned** (VAD-concatenated to ≥30 s clean speech during collection). D+E does
**not** re-run `clean_clip` — it loads the wav and extracts on the whole signal.

---

## 3. Architecture — module layout

New domain lives in new files; existing modules are extended in place following their patterns.

| File | Status | Contents |
|---|---|---|
| `features/ssl_embedding.py` | extend | `extract_layers(signal, sr, layers) -> dict[int, FeatureVector]` — one forward pass, mean-pool each requested layer (reuses the existing model load) |
| `pipeline.py` | extend | `build_segment_features_direct(manifest, extractor) -> tuple[pd.DataFrame, pd.DataFrame]` — load wav → extract on whole signal; **no re-clean, no re-window** |
| `proximity/embedding.py` | extend | add `weighting="channel"` to `language_centroids` (per-channel mean, then per-language mean) |
| `features/aggregate.py` | extend | `aggregate_language_robust(clip_vectors) -> dict[str, float]` — per-feature **median + MAD** (and IQR); mirrors `aggregate_language` |
| `validation/outliers.py` | **new** | `OutlierDetector` ABC, `OutlierResult`, `CentroidMADDetector`, `IsolationForestDetector`, `detect_language_outliers` |
| `validation/robustness.py` | **new** | `proximity_pipeline`, `leave_one_station_out`, `leave_one_segment_out`, `bootstrap_metric_ci` |
| `scripts/embed_segments.py` | **new** | controller **bg job**: XLS-R forward pass per segment → per-layer embedding caches |
| `scripts/build_segment_prosody.py` | **new** | prosody feature table (cheap, synchronous) |
| `scripts/run_data_integrity.py` | **new** | thin analysis driver → `data/data_integrity_results.json` + figures |
| `docs/data-integrity-findings.md` | **new** | the deliverable writeup |

**Analysis form:** tested library functions + a thin driver script. Every headline number traces to
`data/data_integrity_results.json` and every figure to `docs/figures/data-integrity/`. (A notebook
wrapper can be added if the notebook artifact is wanted, but the logic stays in tested modules — this
keeps the analysis subagent-testable, unlike a notebook-only deliverable.)

---

## 4. Feature foundation

### 4.1 Multi-layer SSL extraction (one forward pass)

`SSLEmbeddingExtractor.extract` currently returns **one** layer's pooling (layer is a constructor arg).
Add a method that pools several layers from a **single** forward pass:

```python
def extract_layers(self, signal: np.ndarray, sr: int = 16_000,
                   layers: Sequence[int] = (12, 16, -1)) -> dict[int, FeatureVector]:
    """One forward pass with output_hidden_states=True; mean-pool each requested layer.
    Returns {layer_index: {emb_000: ..., ...}}. Layer indexing matches the constructor `layer`
    arg (index into hidden_states; -1 = last transformer layer)."""
```

The single-layer `extract` stays as-is (used by the prosody-parallel comparison path and by any
per-layer consumer). `extract_layers` is what the caching job calls.

### 4.2 Direct per-segment extraction (bypass clean + window)

`build_segment_features` runs `clean_clip` (VAD) then windows — both wrong for pre-cleaned 30 s
segments. Add:

```python
def build_segment_features_direct(
    manifest: pd.DataFrame, extractor: FeatureExtractor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """manifest requires columns [segment_id, language, channel_id, path] (+ optional source,
    recording_ref carried through to provenance). For each row: load wav (soundfile), extract on the
    WHOLE signal (no clean, no window). Returns:
      prov_df: index=segment_id, cols=[language, channel_id, source, recording_ref]
      feat_df: index=segment_id, cols=extractor feature names."""
```

Provenance uses `channel_id` as the station label (the confound + leave-one-station-out unit).

### 4.3 Embedding cache job (`scripts/embed_segments.py`) — controller background job

- Reads the final manifest, calls `extract_layers(..., layers=(12, 16, -1))` per segment.
- Writes **one parquet per layer**: `data/segment_embeddings_xlsr_l{12,16,last}.parquet`, index
  `segment_id`, columns `emb_000…emb_NNN` (mirrors the existing `clip_embeddings_xlsr_*.parquet`).
- **Single-process sequential** (no thread pool) — the Phase-B segfault came from concurrent CPU torch
  forward passes; this job serializes by construction. ~78 s/clip on CPU (project memory) ⇒ order
  hours; it is a one-time cache.
- **Resumable:** on start, skip `segment_id`s already present in the output parquets; append. A crash
  or interruption resumes rather than restarting.
- `python -u`, progress logged every N segments; launched by the controller as a background task right
  after T1 lands, monitored to completion before T7.

### 4.4 Prosody feature table (`scripts/build_segment_prosody.py`)

`build_segment_features_direct(manifest, ProsodyAcousticExtractor())` →
`data/segment_features_prosody.parquet` (index `segment_id`, 16 scalar columns) +
`data/segment_provenance.parquet` (the shared provenance table). Cheap (CPU, seconds/segment); runs
synchronously as a normal task.

---

## 5. Workstream D — outlier detection

### 5.1 The `OutlierDetector` ABC (mirrors `FeatureExtractor`)

```python
@dataclass(frozen=True)
class OutlierResult:
    scores: np.ndarray       # per-segment outlier score (robust-z, or IF anomaly score)
    is_outlier: np.ndarray   # bool mask, same order as X rows
    threshold: float         # decision boundary used

class OutlierDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def flag(self, X: np.ndarray) -> OutlierResult:
        """X = (n_segments, n_features) for ONE language. Per-segment flags."""
```

Config in the constructor (not in `flag`), exactly as `SSLEmbeddingExtractor` takes `layer`/`pooling`.

### 5.2 Detectors

- **`CentroidMADDetector(threshold=3.5, metric="euclidean")`** — primary; works in **both** spaces.
  Compute the centroid, then each segment's distance to it (`metric="cosine"` on L2-normalized SSL,
  `"euclidean"` on standardized prosody), then **median + MAD** of those distances → robust-z
  `0.6745·(d − median)/MAD`; flag `|z| > threshold`. Robust stats on a **1-D distance** ⇒ stable even at
  1024-dim with n=14–25, where covariance methods fail. (MAD: Leys et al. 2013,
  https://doi.org/10.1016/j.jesp.2013.03.013.) Threshold **3.5** (not 3.0) to avoid over-flagging thin
  es/el. Guard: if `MAD == 0` (≤ half the points identical), fall back to a no-flag result + log.
- **`IsolationForestDetector(contamination="auto", seed=0)`** — swappable 2nd method, **prosody space
  only** (16-dim). Cross-checks MAD flags and demonstrates the ABC. (Liu et al. 2008,
  https://doi.org/10.1109/ICDM.2008.17.) Marked in output with its `name`.

### 5.3 Orchestration

```python
def detect_language_outliers(
    feat_df: pd.DataFrame, labels: dict[str, str],
    detector: OutlierDetector, space: str,   # "prosody" | "ssl" — controls prep
) -> pd.DataFrame:
    """Per language: subset rows, prep the space (prosody→standardize; ssl→L2-normalize),
    run detector.flag. Returns rows [segment_id, language, detector, space, score, is_outlier]."""
```

Run for: (CentroidMAD, prosody), (CentroidMAD, ssl@layer16), (IsolationForest, prosody). Union the flag
tables → `data/segment_outliers.parquet`. Per-language flag counts feed the findings doc as a
data-quality signal (e.g. verification escapes the LLM gate missed).

---

## 6. Workstream E — robust aggregation, the confound re-test, stability, clustering

### 6.1 Robust aggregation & proximity

- **Prosody:** `aggregate_language_robust` (per-feature **median + MAD/IQR**) → per-language table →
  `standardize` → `distance_matrix("euclidean")`. (Replaces the mean+std `build_language_table` path
  for the robust run; the mean+std path is retained for the before/after comparison.)
- **SSL:** `language_centroids(emb_df, weighting="channel")` → cosine distance matrix. Per-**channel**
  weighting (channel mean first, then language mean) stops a channel with 6 segments from dominating a
  language's centroid over one with 1 — a centroid-level confound mitigation. SSL robustness comes from
  outlier exclusion + per-channel weighting, not coordinate-wise median (a coordinate median of unit
  vectors is not a meaningful embedding).

### 6.2 The central test — station vs language confound

Build a **segment-level** distance matrix (178 segments): prosody →
`distance_matrix(standardize(prosody_feat), "euclidean")`; SSL → cosine on L2-normalized segment
embeddings. Feed the existing `confound_report(dist_df, language_labels, station_labels)` with
`language_labels` = segment→language and `station_labels` = segment→`channel_id`. Report
`{language_silhouette, station_silhouette, language_gap, station_gap}` for **both** methods, **against
the Phase-0.5 baseline** (language gap 0.012 / station gap 0.049). **Success = language_gap now ≥
station_gap, or a clear narrowing.** Now non-degenerate (~3 seg/channel).

### 6.3 Method-comparison metrics (per-language proximity)

Reuse `class_silhouette` (rhythm classes via `RHYTHM_CLASS`), `within_between_separation`, and
`mantel_test(proximity, reference_distance_matrix(languages))` vs the Glottolog tree — for baseline and
XLS-R, mirroring `feature-exploration-findings.md` §4 so the before/after is apples-to-apples. Also
`spearman_against_reference` on nPVI where applicable.

### 6.4 Stability (`validation/robustness.py`)

A `proximity_pipeline(feat_df, prov_df, method, exclude=None, weighting=...) -> (dist_df, metrics)`
callback rebuilds the per-language proximity + headline metrics from any **segment subset**, so the
resampling functions just call it:

- **`leave_one_station_out`** — for each `channel_id`, drop its segments, rebuild, recompute
  silhouette/Mantel/confound-gap → distribution across held-out stations (the phase's secondary
  success criterion: quantified station-drop stability).
- **`leave_one_segment_out`** — jackknife over segments.
- **`bootstrap_metric_ci(..., n_boot=1000, seed=0)`** — resample segments **within language** (preserves
  per-language n), rebuild, recompute → percentile CIs on the headline metrics.

All seeded/deterministic.

### 6.5 Report-both

Every headline (confound gaps, silhouette, Mantel r/p, dispersion) is computed on the **full** segment
set and the **outlier-excluded** set; the findings doc shows both plus the delta. Exclusion never runs
silently.

### 6.6 Clustering / visualization

- **Language level (8 nodes), both methods:** `linkage_matrix` → dendrogram; `mds_2d` → 2-D map;
  points colored by `RHYTHM_CLASS`, annotated by `LINEAGE` family. The project's first real "language
  proximity" picture.
- **Segment level (178 nodes):** `mds_2d` of the segment distance matrix, colored **by language** and
  **by `channel_id`** (two panels) — the visual confound story. (UMAP deferred; MDS suffices.)

Figures → `docs/figures/data-integrity/`.

---

## 7. Deliverable — `docs/data-integrity-findings.md`

1. **Coverage table** — segments surviving verification + outlier flags per language; flagged-outlier
   counts per method/space; es/el thin caveat.
2. **Before/after metrics** — rhythm-class silhouette, within/between, Mantel r/p, for baseline + XLS-R,
   vs Phase-0.5 §4; **with/without outlier exclusion** (delta).
3. **Before/after confound** — language vs station gaps vs Phase-0.5's 0.012/0.049 — the headline test.
4. **Stability** — leave-one-station-out spread + bootstrap CIs on the headline metrics.
5. **Interim clustering** — dendrogram + MDS figures, both methods, captioned honestly.
6. **Caveats** — es/el carpet-scarcity + thin n; radio/podcast register; XLS-R still provisional;
   confound check strength; layer-sensitivity note.

---

## 8. Task sequence (one plan, subagent-driven; bg job pipelined)

| # | Task | Kind |
|---|---|---|
| T1 | `extract_layers` (one-pass multi-layer) + `build_segment_features_direct` | TDD, fakes |
| T2 | Launch `embed_segments.py` **bg job** (right after T1) | controller |
| T3 | Prosody table `build_segment_prosody.py` (+ provenance parquet) | synchronous, real extractor |
| T4 | `OutlierDetector` ABC + `CentroidMADDetector` + `IsolationForestDetector` + `detect_language_outliers` | TDD, synthetic |
| T5 | `aggregate_language_robust` + `language_centroids(weighting="channel")` | TDD |
| T6 | `validation/robustness.py` (`proximity_pipeline`, LOSO, LOO, bootstrap CI) | TDD, synthetic |
| T7 | `run_data_integrity.py` — consume cached embeddings (ready by now) + prosody → results JSON + figures | integration driver |
| T8 | Write `docs/data-integrity-findings.md` from the results | controller |

T2's job runs while T3–T6 (the bulk of the TDD work) proceed; the controller confirms the caches exist
before T7. If the job is still running at T7, the controller waits/monitors rather than re-running.

---

## 9. Testing

- **Fakes/synthetic by default** (existing convention, no `conftest.py`; inline fixtures). Fake
  extractors for `build_segment_features_direct`; synthetic feature-frames with planted outliers for
  the detectors; synthetic distance matrices for stability/confound (extends `test_proximity_agreement`
  patterns).
- **Determinism:** all resampling seeded; assert stable outputs.
- **Real-model paths** (`extract_layers` against XLS-R) marked `@pytest.mark.slow`, deselected by
  default (matches `pyproject.toml`).
- Detector edge cases: `MAD == 0` fallback; single-segment language; all-identical inputs.

---

## 10. Reuse map (don't rebuild)

- `features/{base.FeatureExtractor, prosody_acoustic, ssl_embedding}` — extractors + the ABC to mirror.
- `proximity/{distance.standardize|distance_matrix|linkage_matrix|mds_2d, embedding.language_centroids}`.
- `validation/{proximity_agreement.class_silhouette|within_between_separation|confound_report,
  family_tree.reference_distance_matrix|mantel_test|LINEAGE, typology.RHYTHM_CLASS|REFERENCE_NPVI_V|
  class_separation|spearman_against_reference}`.
- `features/aggregate.{aggregate_language, build_language_table}` (robust variant mirrors these).
- Per-length XLS-R caches from Phase 0.5 are for the **old** 55-clip corpus (keyed by clip) — **not**
  reused; the new caches are keyed by `segment_id`.

---

## 11. Success criteria

- **Primary:** the integrity pipeline **reduces the channel confound** (language gap ≥ station gap, or a
  clear narrowing vs 0.012/0.049) and/or **improves** typology/family agreement vs Phase-0.5 —
  reported **before/after** with stability/CIs, not single points.
- **Secondary:** a credible **interim language clustering** (dendrogram + MDS, both methods) with
  quantified leave-one-station-out stability, honestly captioned.

---

## 12. Out of scope (this phase)

Scaling past the 8 seed languages (Phase 2); finalizing the feature method (XLS-R stays provisional);
production ingest hardening (Phase 1); UMAP; Mahalanobis/MCD; source/geo/accent attribution (a future
phase — project memory `future-source-geo-phase`).

---

*Companion documents: `docs/data-integrity-phase.md` (phase brief),
`docs/feature-exploration-findings.md` (Phase-0.5 decision + the confound this phase re-tests),
`docs/superpowers/specs/2026-06-24-music-of-languages-design.md` (design spec §7 robustness/validation).*

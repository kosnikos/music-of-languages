# Feature Exploration Cycle — Findings & Method Decision

**Status:** Cycle complete — SSL (XLS-R) implemented and compared head-to-head against the
alignment-free prosody baseline. Verdict: **no clear winner on current data; data + confound
control is the bottleneck, not the method.**
**Date:** 2026-06-26
**Branch:** `feature-exploration-ssl`
**Author:** Konstantinos (with Claude)

This document is the output of the Feature Exploration cycle (Phase 0.5; see
`docs/feature-exploration-cycle.md`). It **supersedes** the Phase 0 baseline recommendation
(`docs/phase0-findings.md`) and records which feature method(s) carry into Phase 1, with the
comparative evidence. Every number traces to the executed notebook
`notebooks/04_ssl_vs_prosody.ipynb` and the per-length embedding caches
`data/clip_embeddings_xlsr_{None,30.0,10.0}.parquet` (gitignored).

---

## 1. Summary verdict

On an 8-language, 55-clip internet-radio corpus, a multilingual SSL embedding (**XLS-R-300m**)
is **directionally better** than the alignment-free prosody baseline at reproducing rhythm
typology and language-family structure — but the margin is weak, statistically inconclusive,
and partly confounded by recording channel. The precise reading:

> XLS-R shows directionally better rhythm-class separation than the prosody baseline
> (silhouette **+0.046** vs **−0.004**, best configs) and a **positive** genealogical
> association where the baseline shows a **negative** one (Mantel **r = +0.290** vs **−0.103**),
> but **neither result is statistically significant** (Mantel p = 0.076; near-zero absolute
> silhouette for both), both comparisons are **best-of-15-configs vs best-of-3**, the
> **segment-level geometry clusters more strongly by station than by language** (gap 0.049 vs
> 0.012), and the per-recording centroid weighting only **partially** addresses that confound.
> The most defensible reading: XLS-R is **plausibly** capturing some language-level signal, but
> the signal-to-noise ratio on this corpus is **too low to confirm** it is capturing linguistic
> rhythm rather than channel characteristics.

**Decision (see §7):** carry XLS-R forward as the **provisional** leading SSL method (mid layers
~12–16, ~30 s windows), keep the interpretable prosody baseline alongside it, and **do not
finalize** a single method yet. The cycle's real finding is that the comparison is **data- and
confound-limited**; resolving that — deeper/balanced multi-station data, channel controls, and
the deferred robustness/outlier strand — is the prerequisite for a confident choice.

---

## 2. What this cycle built

All behind the existing `FeatureExtractor` / proximity / validation interfaces, TDD'd:

- **`SSLEmbeddingExtractor`** (`features/ssl_embedding.py`) — configurable HF SSL model + layer +
  pooling (wav2vec2 / XLS-R / HuBERT); mean (or mean⊕std) pool of a chosen hidden layer.
- **Embedding proximity path** (`proximity/embedding.py`) — L2-normalized, per-recording-weighted
  per-language centroid → cosine, distinct from the scalar baseline's `mean+std → z-score →
  euclidean` path.
- **Method-agnostic comparison metrics** (`validation/proximity_agreement.py`) — rhythm-class
  silhouette, within/between separation, and a station/channel confound report.
- **Family-tree agreement** (`validation/family_tree.py`) — a Glottolog-derived genealogical
  reference distance matrix + a permutation Mantel test.
- **Windowing + provenance** (`clean/window.py`, `ingest/manifest.py`, `pipeline.py`) —
  configurable analysis-window length with sub-clip → recording → station provenance, so
  aggregation is hierarchical (no pseudoreplication) and the confound check is possible.
- **Light corpus expansion** — the collector now records `clips_manifest.parquet` with station
  provenance.

---

## 3. Corpus coverage (the lens for every number below)

55 usable clips, all 8 seed languages, but **markedly uneven** (from `clips_manifest.parquet`):

| Language | clips | distinct stations | | Language | clips | distinct stations |
|---|---:|---:|---|---|---:|---:|
| french | 15 | 13 | | english | 3 | 3 |
| greek | 15 | 15 | | finnish | 2 | 2 |
| spanish | 13 | 13 | | german | 2 | 2 |
| polish | 4 | 4 | | **italian** | **1** | **1** |

French/Greek/Spanish are well-sampled with strong station diversity; **Italian (1 clip / 1
station), Finnish, German, and English are thin.** The same structural data-sourcing limits
Phase 0 documented recur: German public broadcasters geo-block from the collection host; Finnish
(Yle) and BBC English are served over HLS/`.m3u8` the simple `ffmpeg` path can't capture; Italian
talk stations near Rome are sparse. Fixing these (HLS capture, geo-fallback) is **Phase-1 scope**.
Thin languages make their per-language centroids and rhythm-class silhouettes **high-variance**.

The head-to-head remains valid because it is **relative**: both methods see the **same** clips.

---

## 4. Method comparison (the head-to-head)

Per-language proximity → rhythm-class **silhouette** (↑ = classes better separated),
**within/between gap** (↑ = separated), and **Mantel r** vs the Glottolog tree (↑ = more
genealogical agreement). Baseline: scalar prosody → standardize → euclidean. SSL: centroid →
cosine. (`notebooks/04`, cells 4–5.)

**Prosody baseline** (reconfirms the Phase-0 weak result — no class or family recovery):

| length | silhouette | Mantel r | Mantel p |
|---|---:|---:|---:|
| 10 s | −0.004 | −0.103 | 0.625 |
| 30 s | −0.052 | −0.217 | 0.839 |
| full | −0.090 | −0.284 | 0.928 |

**XLS-R-300m** (best layer per length shown; full layer sweep {8,12,16,20,last} in the notebook):

| | length | layer | silhouette | Mantel r | Mantel p |
|---|---|---:|---:|---:|---:|
| best silhouette | 30 s | 16 | **+0.046** | 0.054 | 0.432 |
| best Mantel r | full | last | −0.270 | **+0.290** | 0.076 |
| (mid layer, full) | full | 16 | +0.002 | 0.140 | 0.280 |

**Readings:**
- **XLS-R edges the baseline on both headline metrics** — silhouette +0.046 vs −0.004; Mantel
  +0.290 (positive) vs −0.103 (negative). It is also directionally better across *most* configs,
  not only at the cherry-picked best (see §6 caveat 1).
- **Layer matters:** mid layers (12, 16) give the best rhythm-class silhouette; the **last** layer
  is worst for silhouette but best for the family-tree Mantel — consistent with mid layers
  carrying more phonetic/prosodic substrate and later layers more pretraining-objective structure.
- **Length matters:** 30 s gives the best XLS-R rhythm-class silhouette; full gives the best Mantel.
  10 s is generally worst for SSL separation.
- **But the absolute separations are near zero** and **the best Mantel is not significant**
  (p = 0.076). Neither method *strongly* recovers typology or family structure on this corpus.

---

## 5. Channel / station confound check (the critical caveat)

Cycle brief requirement: do embeddings cluster by **language** or by **channel**? For the best
XLS-R config, on a **segment-level** cosine distance matrix (48 segments, 45 stations, 42 of them
singletons) with language vs station labels (`notebooks/04`, cell 5):

| | silhouette | within/between gap |
|---|---:|---:|
| by **language** | −0.057 | 0.0125 |
| by **station** | −0.028 | **0.0491** |

**Station separation exceeds language separation** (gap 0.049 vs 0.012; station silhouette also
less negative). i.e. in the raw segment geometry the embedding clusters **at least as much by
recording channel as by language** — exactly the failure mode the confound check exists to catch.

**Precision (important):** this is a **segment-level** result. The headline comparison (§4) uses a
**per-recording-weighted centroid**, which already partially guards against a chatty station
dominating a language; the confound check deliberately bypasses that guard to test the raw
geometry (stations are not represented at the centroid level). So the correct statement is: *the
segment-level geometry is station-confounded; the per-recording centroid weighting mitigates this
to a degree this notebook does not quantify.* The confound check is itself **weak** — with 42/45
stations contributing a single clip, station silhouette is near-degenerate, so this is
**directional evidence**, not a measured effect. It is, however, enough to **withhold confidence**
in the SSL "win."

---

## 6. Threats to validity (the decision doc's honest caveats)

1. **Best-of-configs selection.** XLS-R's best is chosen over 5 layers × 3 lengths = **15 configs**;
   the baseline's best over **3**. On an 8-node distance matrix a 0.05 silhouette gap is an
   **ordering** result, not an effect size. XLS-R is directionally better across most configs, which
   is the defensible claim — not the magnitude of the single best.
2. **Statistical significance.** Best Mantel p = 0.076 (not significant at 0.05); the Glottolog
   reference is a coarse depth-based 8×8 tree. Report as "positive vs negative association, neither
   significant," never "agrees with the family tree."
3. **Segment vs centroid confound geometry** — see §5.
4. **Thin-language aggregation artifacts.** Italian (n = 1 clip) has per-feature std = 0, so its
   baseline `_std` columns are constant-zero — a non-meaningful representation in standardized
   space (zero dispersion, not missing). Other thin languages (n = 2–3) have unstable std.
5. **Baseline aggregates per-segment, not per-clip.** At 10 s a long clip contributes many windows
   that vote equally in the per-language mean/std, while the SSL path averages per-recording then
   per-language (two-level). This asymmetry slightly biases the comparison **against** the baseline
   at 10 s.
6. **Window drop-out.** Clips with < 30 s of clean speech produce no 30 s window and drop from the
   30 s rows (the `[skip-base]` skips — expected, not a bug); both methods drop the same clips.
7. **Within-recording stability** (npvi_v std 8.25 at 10 s → 6.06 at 30 s ⇒ longer windows more
   stable) is a **prosody-scalar** signal; it does **not** speak to SSL embedding stability.

---

## 7. Decision — what carries into Phase 1

**No method is finalized.** The evidence does not support declaring XLS-R the method on current
data. Concretely:

- **Carry XLS-R forward as the provisional leading SSL method** — mid layers (~12–16), ~30 s
  windows, mean-pool, cosine on per-recording-weighted centroids. It is directionally the best
  candidate, inference-only/CPU-friendly, and beats the baseline where the baseline is flat or
  inverted. But it is **provisional**, pending the data + confound work below.
- **Keep the prosody baseline** in the harness as the interpretable reference and a sanity anchor;
  it remains weak (reconfirmed) but explainable.
- **The cycle's headline conclusion is that the comparison is data- and confound-limited**, not
  that a method was chosen. The bottleneck is the corpus and channel confound, not the extractor.

---

## 8. Recommendations for Phase 1 / the next cycle

Directly motivated by the findings above:

1. **Deeper, balanced, multi-station data.** Tens of clips per language across **several distinct
   stations and times** — the single biggest lever on every number here. Fix the structural gaps:
   **HLS/`.m3u8` capture** (recovers Finnish/BBC-English), **geo-fallback/mirrors** (German),
   broader Italian sourcing.
2. **Channel-confound controls.** The segment-level station > language signal (§5) is a red flag.
   Phase 1 needs an explicit confound protocol: same-language-across-stations vs
   different-language-same-channel, codec/normalization controls, and the centroid-level mitigation
   **quantified** (not assumed).
3. **Run the deferred robustness/outlier strand** (the cycle the user sequenced second) — outlier
   filtering (MAD / distance-from-centroid), robust aggregation (median), leave-one-clip-out
   stability, full clip-length sensitivity. The thin-data variance and channel confound here make
   this strand clearly necessary, and it is what would turn "directional" into "confident."
4. **Per-clip (not per-segment) baseline aggregation** option, to remove the §6.5 asymmetry, and
   imputation rather than column-drop for thin-language std artifacts (§6.4).

---

## 9. Deferred (out of scope this cycle)

VoxLingua107 **ECAPA** (purpose-built language-ID "upper bound"), **envelope-RNN**, **MFA** rhythm
metrics, **fusion** of methods, and the full robustness/outlier **Validator ABC** — all remain
candidates for subsequent cycles. The harness (extractor + proximity + validation interfaces) is
ready to accept them.

---

*Companion documents: `docs/phase0-findings.md` (baseline, superseded by this decision),
`docs/feature-exploration-cycle.md` (cycle brief),
`docs/superpowers/specs/2026-06-25-feature-exploration-cycle-design.md` (this cycle's design),
`notebooks/04_ssl_vs_prosody.ipynb` (the comparison; re-run to regenerate figures).*

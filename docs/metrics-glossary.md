# Metrics & Statistics Glossary

A plain-language companion to `docs/data-integrity-findings.md`. Every statistical term used there —
*language gap, silhouette, rhythm silhouette, Mantel r, LOSO median, bootstrap CI, …* — gets a short
section: what it is, a tiny worked example, how it's used in this project, and a link when one is
especially clarifying.

The whole analysis works on **distance matrices**: an *N × N* table where entry *(i, j)* is how far
apart items *i* and *j* are (0 on the diagonal, symmetric). The items are either the **8 languages**
(per-language proximity) or the **178 segments** (the confound check). Everything below either builds
such a matrix, scores structure in it, or estimates how much to trust that score.

---

## A. How "distance" is measured

### A1. Standardization (z-score)
Rescale each feature to mean 0, standard deviation 1: `z = (x − mean) / std`. Without it, a feature
measured in big units dominates a straight-line distance.
- **Example:** f0 in Hz (mean ≈ 150, std ≈ 30) and nPVI (mean ≈ 50, std ≈ 10) live on different
  scales; after z-scoring both are unitless and comparable, so neither swamps the other.
- **Here:** the 16 prosody scalars are z-scored before Euclidean distance (`standardize` →
  `distance_matrix(..., "euclidean")`).
- **Link:** https://en.wikipedia.org/wiki/Standard_score

### A2. Euclidean distance (used for prosody)
Ordinary straight-line distance, `sqrt(Σ (xᵢ − yᵢ)²)`, on the standardized features. Sensitive to
magnitude, which is why A1 comes first.

### A3. Cosine distance (used for SSL embeddings)
`1 − cos(angle)` between two vectors — it compares **direction**, ignoring length.
- **Example:** `[1, 0]` vs `[0, 1]` → angle 90° → cosine 0 → distance **1** (maximally different
  direction). `[1, 0]` vs `[2, 0]` → same direction → distance **0** (length ignored).
- **Here:** XLS-R embeddings are L2-normalized and compared by cosine, the standard way to compare
  neural speech/speaker embeddings (loudness/scale shouldn't matter; the *direction* in embedding
  space carries the content).
- **Link:** https://en.wikipedia.org/wiki/Cosine_similarity

---

## B. How items are summarized into a per-language point

### B1. Per-recording / per-channel weighted centroid
A "centroid" is the average vector of a group. A **weighted** centroid first averages *within* a
sub-group, then averages those, so a group that contributed many segments can't dominate.
- **Example:** channel X gave 3 segments near `[1, 0]`, channel Y gave 1 near `[0, 1]`. A flat
  average is `[0.75, 0.25]` (X wins by count). Averaging **per channel first** gives
  `mean([1,0], [0,1]) = [0.5, 0.5]` — X and Y count equally.
- **Here:** language centroids use **per-channel** weighting so a talkative station doesn't re-inject
  the very channel confound we're trying to remove.

### B2. Median, MAD, IQR (robust aggregation)
Robust = resistant to a few extreme values.
- **Median** — the middle value (50th percentile).
- **MAD** (median absolute deviation) — `median(|x − median|)`, a robust spread.
- **IQR** (interquartile range) — `p75 − p25`, another robust spread.
- **Example:** `[50, 52, 54, 1000]` → **mean 289** and **std ≈ 410** are wrecked by the outlier, but
  **median 53** and **MAD 2** barely move.
- **Here:** each language's prosody summary uses median (+ MAD/IQR as dispersion) instead of
  mean/std, so one odd segment can't shift the language's position.
- **Link:** https://en.wikipedia.org/wiki/Median_absolute_deviation · MAD as an outlier rule: Leys et
  al. 2013, https://doi.org/10.1016/j.jesp.2013.03.013

---

## C. How separation is scored

### C1. Within/between "gap"
Split all pairs of items into **within-class** (same label) and **between-class** (different label),
average each distance, and take `gap = between_mean − within_mean`. A larger gap ⇒ classes sit
farther apart than members of the same class ⇒ better separation.
- **Example:** points A₁=0.0, A₂=0.1 (class A) and B₁=10.0, B₂=10.1 (class B). Within ≈ 0.1, between ≈
  10.05, **gap ≈ 9.95** — strongly separated.
- **Here:** `language gap` uses the language labels; `station gap` uses the `channel_id` labels — on
  the **same** distance matrix (see C4).

### C2. Silhouette
For each item: `a` = its mean distance to its **own** class, `b` = its mean distance to the **nearest
other** class; its silhouette is `s = (b − a) / max(a, b)`. Average over items.
- Range **[−1, 1]**: `> 0` the item sits with the right group; `≈ 0` on a boundary; `< 0` it is
  actually closer to another class (mis-clustered).
- **Example:** two tight, far-apart blobs → silhouette near **+1**; overlapping blobs → near **0**;
  labels essentially random → slightly negative.
- **Here:** `language silhouette` (labels = language) and `station silhouette` (labels = channel)
  summarize how cleanly the geometry clusters by each.
- **Links:** Rousseeuw 1987, https://doi.org/10.1016/0377-0427(87)90125-7 ·
  https://en.wikipedia.org/wiki/Silhouette_(clustering) ·
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

### C3. Rhythm silhouette
The **same** silhouette computation (C2), but the labels are each language's **rhythm class** —
*stress-timed* (e.g. English, German, Polish), *syllable-timed* (French, Spanish, Italian), or
*intermediate* (Greek, Finnish). Computed on the 8-language proximity.
- **Meaning:** `> 0` means the feature geometry groups stress-timed languages together and
  syllable-timed together — i.e. it **recovers rhythm typology**. This is the linguistics payoff, not
  just clustering hygiene.
- **Here:** SSL rhythm silhouette 0.095 (positive) vs prosody 0.057 — SSL recovers rhythm structure a
  bit better; both are weak in absolute terms (only 8 nodes).

### C4. The confound check (same geometry, two labelings) and the gap ratio
Score the identical distance matrix twice — once with language labels, once with channel labels. If
the geometry separates **by station at least as well as by language**, the "language signal" may be
partly a recording-channel artifact. A compact summary is the **gap ratio** `station_gap /
language_gap`.
- **Example / here:** Phase 0.5 had station gap 0.049 vs language gap 0.012 → ratio **3.9×** (station
  dominated). D+E: 0.029 vs 0.021 → ratio **1.4×**, and the silhouette *flips* (language 0.273 >
  station 0.193). That shrinkage is the phase's central result.

---

## D. Agreement with the family tree

### D1. Mantel r and its permutation p-value
The **Mantel test** measures how strongly two distance matrices agree, entry-by-entry — here, the
data-derived language proximity vs a **Glottolog** genealogical-tree distance (how far apart two
languages are on the family tree).
- **Mantel r** is just a correlation of the two matrices' off-diagonal entries, in **[−1, 1]**: `+1`
  = languages close on the family tree are close in the data; `0` = no relationship; negative =
  inverted.
- **Why a permutation p, not an ordinary one:** matrix entries share items, so they aren't
  independent and normal significance formulas don't apply. Instead you **shuffle** the language
  labels many times, recompute r each time, and ask *how often a random labeling beats the observed
  r*; that fraction is the p-value.
- **Example / here:** SSL Mantel r = 0.128 (weak positive genealogical agreement), permutation p =
  0.24 — i.e. random labelings reach ≥ 0.128 about a quarter of the time on only 8 languages, so it
  is *not* significant at 0.05 even though the value is positive.
- **Link:** https://en.wikipedia.org/wiki/Mantel_test

---

## E. Outlier detection (Workstream D)

### E1. Robust distance-from-centroid (CentroidMAD)
For each language, compute every segment's distance to the language centroid, then flag segments
whose distance is an extreme **robust z-score** — `z = 0.6745 · (d − median(d)) / MAD(d)` — above a
threshold (3.5). Only the far tail (large `z`) is flagged, never points *near* the centre. The
`0.6745` puts MAD on the same scale as a standard deviation for normal data.
- **Example:** distances `[0.9, 1.0, 1.1, 1.0, 8.0]` → median 1.0, MAD ≈ 0.1; the `8.0` is dozens of
  robust-z out → flagged; the rest are not.
- **Here:** run in both the prosody (euclidean) and SSL (cosine) spaces; conservative (flagged 5 and
  7 of 178).

### E2. Isolation Forest and `contamination`
An ensemble that repeatedly splits the data at random; points that get **isolated in very few splits**
are anomalies (they sit in sparse regions). `contamination` is the assumed outlier fraction used to
set the cutoff.
- **Here:** `contamination="auto"` guessed **27 %** of prosody segments were outliers — far too many;
  excluding them *hurt* the metrics. Lesson in the findings: robust aggregation (B2) already handles
  anomalies, so a blunt model-based gate isn't needed.
- **Links:** Liu et al. 2008, https://doi.org/10.1109/ICDM.2008.17 ·
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

---

## F. How much to trust a number (uncertainty & stability)

### F1. Bootstrap confidence interval (CI)
Resample the data **with replacement** many times (here 1000), recompute the statistic on each
resample, and take the 2.5th–97.5th percentiles of those values → a **95 % CI**. It answers "how much
would this number wobble if we'd drawn a slightly different sample?"
- **"Excludes zero"** = the entire interval is on one side of 0, so the effect is unlikely to be a
  fluke of the particular sample.
- **Example / here:** SSL rhythm silhouette 0.095 with CI **[0.048, 0.140]** — wholly above 0, so a
  positive result that survives resampling; prosody's CI **[−0.070, 0.120]** straddles 0, so
  indistinguishable from nothing.
- **Link:** https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

### F2. Leave-one-station-out (LOSO), and the LOSO median
A jackknife over channels: drop **all** segments from one channel, recompute the metric, and repeat
for each of the 59 channels → a distribution reported as **min / median / max**. It checks whether any
single station is secretly driving the result. The **LOSO median** is the typical value across those
refits.
- **Example / here:** SSL rhythm silhouette LOSO `[0.076, 0.095, 0.104]` — tight and always positive,
  so no one station carries it; the SSL Mantel r LOSO `[0.100, 0.209]` never crosses 0 (robust),
  whereas prosody's `[−0.043, 0.129]` does (fragile).

### F3. Permutation test vs bootstrap — why they can seem to "disagree"
They answer different questions, so a positive bootstrap CI *and* a non-significant permutation p is
not a contradiction:
- **Permutation** (D1) tests a **null of no structure**: "could random labelings produce this?"
- **Bootstrap** (F1) estimates the **uncertainty of the estimate**: "how much does it wobble under
  resampling?"
- On only 8 languages, random tree-labelings hit r ≥ 0.128 fairly often (p = 0.24), yet the estimate
  itself is stably positive across resamples (CI excludes 0). Both statements are true and reported.
- **Link:** https://en.wikipedia.org/wiki/Permutation_test

### F4. Pseudoreplication (the reason for "one segment per recording")
Treating **correlated** measurements as if they were independent — e.g. many 30 s windows cut from the
*same* recording (same speaker, mic, codec) — which fakes a larger sample and inflates confidence.
- **Here:** taking exactly **one segment per recording** removes it, so the 178 segments are genuinely
  independent and the confound/stability numbers are honest.
- **Link:** https://en.wikipedia.org/wiki/Pseudoreplication (Hurlbert 1984,
  https://doi.org/10.2307/1942661)

---

## G. How the pictures are made (visualization)

### G1. Multidimensional scaling (MDS)
Places the items in 2-D so that their on-screen distances reproduce the high-dimensional distance
matrix as faithfully as possible — a faithful *map* of the distances, purely for the eye.
- **Here:** the language MDS (8 points) and the segment MDS (178 points, coloured by language vs
  channel) in §6 of the findings.
- **Link:** https://en.wikipedia.org/wiki/Multidimensional_scaling

### G2. Hierarchical clustering, Ward linkage, and the dendrogram
Repeatedly merge the two closest clusters until one remains; the **dendrogram** is the resulting tree,
where the **height** of each merge = how dissimilar the merged groups were. **Ward linkage** chooses,
at each step, the merge that increases within-cluster variance the least (tight, balanced clusters).
- **Reading it:** languages joined low in the tree are more similar; a tall join means a big jump.
- **Links:** https://en.wikipedia.org/wiki/Ward%27s_method ·
  https://en.wikipedia.org/wiki/Hierarchical_clustering

---

*Companion to `docs/data-integrity-findings.md` (the results these metrics describe) and
`docs/feature-exploration-findings.md` (Phase 0.5, which introduced the silhouette / Mantel / confound
comparison). Implementations: `src/musiclang/proximity/` (distances, MDS, linkage),
`src/musiclang/validation/` (silhouette, within/between, confound_report, Mantel, outliers,
bootstrap/LOSO).*

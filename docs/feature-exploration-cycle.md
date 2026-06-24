# Feature Exploration Cycle (Phase 0.5) — Brief

**Purpose:** the dedicated research round that runs **after Phase 0** (which builds the pipeline +
harness + a *validated baseline* method) and **before Phase 1** (which hardens the chosen method
into the end-to-end pipeline). **This is where the real feature-method decision is made** — by
empirically implementing and comparing multiple methods, not just the Phase 0 baseline.

This is a brief/outline only. When the cycle starts it gets its own full
**brainstorm → spec → plan → execute** pass.

## Why this exists

Phase 0 only *empirically* validates ONE method (alignment-free hand-crafted prosody) and
paper-reviews the rest. Choosing among methods honestly requires comparing real implementations.
Pretrained **SSL embeddings** in particular are cheap to try (inference-only, no training) and may
outperform — or usefully complement — hand-crafted prosody. This cycle does that comparison
properly instead of deciding on a single empirical data point.

## Prerequisites (what Phase 0 leaves you)

- The `FeatureExtractor` interface (`src/musiclang/features/base.py`).
- A validated baseline extractor (`ProsodyAcousticExtractor`).
- Proximity utilities (`src/musiclang/proximity/distance.py`) and typology validation
  (`src/musiclang/validation/typology.py`).
- A working radio sample collector (`scripts/collect_sample.py`) + a small per-language dataset.
- `docs/phase0-findings.md`: baseline results + the recommended agenda for this cycle.

## Candidate methods to implement & compare

All behind the same `FeatureExtractor` interface; all evaluated with the same harness.

1. **Hand-crafted prosody (baseline)** — already built in Phase 0. The reference to beat.
2. **Pretrained SSL embeddings (priority — cheap, inference-only):**
   - XLS-R / wav2vec 2.0 / HuBERT — mean/attentive-pool a chosen hidden layer.
   - VoxLingua107 ECAPA spoken-language-ID embeddings — purpose-built for language identity.
   - Whisper encoder embeddings.
   Compare layers/pooling; cosine distance for the proximity matrix.
3. **Learned envelope embedding (Deloche-style RNN)** — heavier (needs training on collected
   audio). Reproduce the amplitude-envelope + voicing approach; use network activations as the
   rhythm fingerprint. Optional/stretch within this cycle.
4. **Forced-alignment rhythm metrics (MFA)** — heavy (per-language models). Test whether true C/V
   segmentation materially improves on the alignment-free %V/nPVI. Optional/stretch.
5. **Fusion** — combine the best hand-crafted + learned features (concatenate standardized vectors,
   or combine distance matrices) and test whether it beats either alone.

## Evaluation protocol (how to compare them)

For each method, using the shared harness:
- Build the per-language proximity matrix.
- **Typology agreement** — class separation + Spearman vs reference nPVI; dendrogram/MDS by class.
- **Family-tree agreement** — correlate the proximity matrix against a known language-relatedness
  reference (Glottolog tree distances or lexical ASJP distances) via a Mantel test. (Prosody
  similarity need NOT match genetic family — divergence is itself an interesting finding — but the
  comparison is a useful sanity anchor.)
- **Robustness** — dispersion across clips/speakers/stations; sensitivity to clip length.
- **Interpretability / blog-friendliness** — can we explain what drives the similarity?

## Data needs

Likely more than Phase 0's tiny sample — enough clips/hours per language to stabilize estimates
and to expose channel/speaker confounds. Reuse the Phase 0 collector with larger N and more
stations per language.

## Confound checks (critical for learned methods)

Learned embeddings can encode codec/channel/speaker/topic rather than "music." Explicitly test:
do embeddings of the *same language from different stations* cluster together more than *different
languages from similar channels*? If channel dominates, the method is not measuring language sound.

## Gate / decision (the cycle's output)

A decision doc recording **which method(s) or fusion to carry into Phase 1**, with comparative
evidence on typology agreement, family-tree agreement, robustness, and confound-resistance. This
supersedes the Phase 0 baseline recommendation and finalizes the `features`-stage shape for Phase 1
(see `docs/phase1-handoff.md`).

## Deliverables

- New `FeatureExtractor` implementations in `src/musiclang/features/` (at least the SSL one).
- Method-comparison notebooks.
- The decision doc (e.g. `docs/feature-exploration-findings.md`).

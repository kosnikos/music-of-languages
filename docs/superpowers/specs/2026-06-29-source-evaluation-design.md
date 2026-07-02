# Workstream A — Source Evaluation — Design Spec

**Phase:** Data Integrity (0.75) · **Workstream:** A (Alternative clip-source research) · **Date:** 2026-06-29
**Branch:** `data-integrity-brief` · **Author:** Konstantinos (with Claude)

This spec covers **Workstream A only**. Workstreams B–E (independent segments, LLM verification,
outlier detection, robust aggregation) get their own brainstorm → spec → plan cycles within the
phase. Companion documents: `docs/data-integrity-phase.md` (the phase brief — Workstream A is
"do this FIRST"), `docs/feature-exploration-findings.md` (the confound motivation),
`docs/capital-stations-findings.md` (the metadata language-ID ceiling),
`docs/superpowers/specs/2026-06-24-music-of-languages-design.md` (design spec; §7 robustness, §9
conventions).

---

## 1. Summary

Workstream A is a **research/evaluation** task, not a collection task. Internet radio via
radio-browser + progressive streams hit structural limits in Phase 0/0.5 (German geo-block, Finnish
Yle + BBC English over HLS, sparse Italian, music/jingle leakage). Before investing in segmentation
(B), verification (C), outlier work (D), or robust aggregation (E), we survey and empirically probe
alternative/complementary sources for clean, attributable, target-language **speech** — judged by
their fitness for the phase's analysis unit: **one independent ~30 s clean-speech segment per
recording, across many distinct channels per language, for all 8 seed languages.**

The output is a **source-evaluation matrix** + a **recommended source mix** (likely per-language) +
a **proposed segment budget for Workstream B**, written to `docs/source-evaluation.md`.

**Seed languages (8, fixed this phase):** English, German, Polish, French, Spanish, Italian, Greek,
Finnish.

## 2. Decisions locked in the brainstorm

These four were settled with the user and are not relitigated below; the rest of the spec builds on them.

1. **Evaluation depth = hybrid.** Desk-survey all source families against the criteria, then run
   **small real capture probes** on the top 2–3 finalists to measure the criteria that only surface
   when you try. (Full per-segment validation — LLM/ASR — is **Workstream B/C**, not A; A's probes
   measure capturability + cleanliness + independence only.)
2. **Register stance = natural primary, corpus as anchor.** Spontaneous-prosody sources (radio talk,
   conversational/interview podcasts) are the **primary** register because rhythm/prosody is the
   target signal and read register distorts it. **One** read/parliamentary corpus is included as a
   **high-integrity anchor/control**, with its register difference explicitly flagged and analyzed —
   never blended in silently.
3. **Candidate families = four.** Improved radio capture (+ extra aggregators), podcasts, broadcaster
   on-demand/news APIs, and one public corpus anchor. **YouTube / large-but-messy web is excluded**
   (licensing murky; independence/attribution unenforceable).
4. **Probes cover all 8 seed languages** (not a representative subset). The Phase-0 gaps are
   language-specific, so per-language probe numbers are what make the recommendation evidence-based.

## 3. Goals & non-goals

**Goals**
- Produce a **source-evaluation matrix** (sources × criteria) with a **per-language coverage
  sub-table**, backed by desk research and finalist probe measurements.
- Recommend a **source or mix** (per-language where coverage demands it) that maximizes independent,
  verified-able, clean target-language speech across many channels per language.
- Propose a concrete **Workstream B segment budget** (independent segments/language, minimum distinct
  channels/language) grounded in what the probes show is actually capturable.
- Decide **how much capture infra to build now vs defer** — i.e. does a lighter source yield enough
  independent verified segments, or must B/Phase-1 build HLS/podcast/corpus fetchers?

**Non-goals (Workstream A)**
- Full collection at the B segment budget (that is Workstream B).
- LLM/ASR per-segment verification (Workstream C).
- Outlier detection (D), robust aggregation + the confound re-run (E).
- Production capture hardening — scheduling, retries, retention, scaling to 30–40 languages (Phase 1).
- Finalizing the feature method (XLS-R stays provisional, per the Phase-0.5 decision).

## 4. Candidate source set

Four families. Each gets a matrix row (or per-instance rows) and a per-language coverage entry. The
research subagents enumerate concrete instances; the items below are the starting set, not exhaustive.

### 4.1 Improved radio capture *(primary, natural)*
- **HLS/`.m3u8` capture** via `streamlink` (preferred) or ffmpeg's HLS path — recovers the streams
  the simple `ffmpeg` progressive path couldn't (Finnish Yle, BBC English).
- **Extra aggregators** beyond radio-browser: Icecast/Shoutcast directory listings, per-country
  public-broadcaster station indexes, to deepen distinct-channel diversity per language.
- Builds on the existing `ingest/radio.py` (`find_stations`, `find_capital_stations`, `record_clip`)
  and the documented radio limits in `docs/phase0-findings.md` / `docs/capital-stations-findings.md`.

### 4.2 Podcasts *(primary, natural, breadth)*
- **RSS / podcast directories** (Podcastindex.org API; Apple Podcasts directory) — talk-heavy shows,
  per-episode downloadable enclosures, many distinct speakers/sessions → strong for **breadth** and
  **independence** (one segment per episode = one per recording, trivially).
- Per-language: identify a handful of talk/interview/news shows per seed language.

### 4.3 Broadcaster on-demand / news APIs *(natural; targets Phase-0 gaps)*
- BBC Sounds, Yle Areena, RAI, ARD/ZDF mediatheken, Radio France, etc. — **where ToS permits**.
- Natural register, well-attributed; directly targets the gap languages (Finnish, German, Italian,
  English). Cost: each is bespoke integration and coverage is uneven across the 8.

### 4.4 Public corpus anchor *(read/parliamentary control, register flagged)*
- **Leading candidate: VoxPopuli** (Wang et al. 2021, arXiv:2101.00390) — covers **all 8** seed
  languages, **spontaneous** parliamentary speech (better register match than read corpora), aligned
  + attributed, permissive license (CC0).
- Compared in the survey against **Common Voice** (read prompts — worse register; CC0) and **MLS**
  (**fails coverage** — no Greek, no Finnish). The survey confirms VoxPopuli or substitutes with
  evidence; exactly **one** anchor is recommended.

## 5. Evaluation criteria (operationalized)

The brief's seven criteria, each made concrete. **✎ = desk-assessable** (research subagents);
**⚙ = measured in the finalist probe**.

| # | Criterion | How assessed | Where |
|---|---|---|---|
| 1 | **Coverage** | Which of the 8 seed langs the source serves; extensibility toward 30–40. Fills the per-language sub-table. | ✎ |
| 2 | **Cleanliness** | % speech via the existing `clean/vad.py` gate on probe samples; observed music/ad/jingle leakage. | ⚙ |
| 3 | **Independence** | Are N fetches genuinely distinct recordings/speakers/channels vs a few long streams? Count distinct channels/shows hit. | ⚙ |
| 4 | **Register match** | Spontaneous vs read; weighted by the natural-primary stance. | ✎ |
| 5 | **Capturability** | Can we programmatically obtain ≥30 s clean speech per recording? HLS? auth? rate limits? Success rate. | ⚙ |
| 6 | **Channel diversity** | Distinct channels/codecs per language available (the lever for *testing* the confound, not baking it in). | ✎/⚙ |
| 7 | **Legal/ToS/attribution** | Recorded per source (deferred-but-informed; exclude clearly-infringing). License + attribution note. | ✎ |

## 6. Method (two-tier)

**Tier 1 — Desk survey (all four families).** Research subagents rate every family against the ✎
criteria, enumerate concrete per-language instances, and fill the coverage sub-table. Output:
structured matrix rows + a per-source legal/attribution note.

**Tier 2 — Finalist probes (top 2–3 finalists, all 8 languages).** For each finalist, fetch a small
number of recordings **per seed language** (target a handful per language — enough to estimate
capture success rate and distinct-channel count, not a full sample) and measure the ⚙ criteria
empirically:
- **capturable** — got a file of the target duration (bool) and capture success rate per language;
- **clean_speech_s** — seconds of clean speech via `clean/vad.py`;
- **meets_30s** — `clean_speech_s ≥ 30`;
- **distinctness** — recording/show/station identifier, to confirm fetches are independent;
- per-language aggregates: success rate, median `clean_speech_s`, # distinct channels reached.

Probes are **selection-focused and throwaway-quality** — no LLM verification (that's C), no robust
aggregation (that's E). They exist to turn the ⚙ criteria from estimates into measured numbers.

## 7. Decision rule

**Hard gates** (a source must pass all to be recommendable):
1. Covers — or is clearly extensible to — the 8 seed languages.
2. Programmatically **capturable to ≥30 s clean speech per recording** (measured in the probe).
3. Yields **many independent recordings** (not a few long streams).
4. **Legal posture acceptable** for research use (deferred-but-informed; clearly-infringing excluded).

**Ranking of survivors:** rate each on the seven criteria **High/Med/Low** (transparent qualitative
ratings — the candidate set is small, so numeric weighting would be false precision). The natural-
primary register stance and independence/channel-diversity are the heaviest factors.

**Output a recommended mix**, expected to be **per-language** because the gaps are language-specific —
e.g. *podcasts for breadth across most languages + broadcaster-API/HLS radio for the Phase-0 gap
languages + VoxPopuli as the clean cross-register anchor.* Plus:
- a **proposed Workstream B segment budget**: target independent segments/language and minimum
  distinct channels/language, grounded in probe-measured capture rates;
- a **build-now-vs-defer** verdict per the §3 goal: which fetchers (HLS, podcast RSS, corpus client)
  B must build vs which defer to Phase 1.

## 8. Components to build (minimal, behind existing stage boundaries)

Intentionally **light** — these are probes, not the Phase-1 ingest package. They reuse the existing
stage boundaries so anything worth keeping hardens in place later.

- **Source adapters** — a thin, common shape per source so the probe runner is source-agnostic:
  - `list_recordings(language, n) -> list[RecordingRef]` (a RecordingRef carries source, channel/show
    id, capture handle/url, and metadata);
  - `capture(ref, out_path) -> Path | None` (writes 16 kHz mono wav; `None` on failure).
  - Adapters: **radio** (reuse `ingest/radio.record_clip`; add an **HLS/streamlink** path), **podcast**
    (RSS fetch + enclosure download via `feedparser`), **corpus** (**VoxPopuli** access via HF
    `datasets`). This is a deliberately minimal echo of the project's interface-driven style, **not**
    the Phase-1 ingest abstraction.
- **Probe runner** — `scripts/probe_source.py`: given an adapter + the 8 languages, fetch the small
  per-language sample, apply `clean/vad.py`, and record results (the §6 fields) to a probe-results
  parquet (`data/source_probe_results.parquet`, gitignored). Controller-driven (see §9).
- **Dependencies** to add via `uv`: `streamlink`, `feedparser`, and HuggingFace `datasets` (for
  VoxPopuli). ffmpeg is installed but **not on the tool-shell PATH** — prepend the winget bin in any
  shell call that invokes it.
- **Provenance (design spec §9):** every probe/adapter records source provenance; the write-up cites
  each source (VoxPopuli: Wang et al. 2021 arXiv:2101.00390; Whisper is a C concern, not A).

## 9. Execution model (subagent-driven)

1. **Research (parallel).** One subagent per source family (4 — independent, no shared state →
   dispatch together), each returning **structured matrix rows**: per-language coverage, ✎ criteria
   ratings, concrete instances, capturability notes (HLS? auth? rate limits?), and a legal/attribution
   note. Subagents have web access (WebSearch/WebFetch).
2. **Synthesis (controller).** Merge rows into the matrix + coverage sub-table; apply the §7 hard
   gates; pick the 2–3 finalists.
3. **Probe (controller-driven, TDD'd adapters).** Build the adapters + probe runner test-first; run the
   captures from the **controller** as a background job, **not inside a subagent** (per the project
   memory note on long-running capture/inference). Real network fetches are integration-checked
   separately from unit tests (which use fakes/fixtures).
4. **Write-up (controller).** Compose the matrix + recommended mix + B segment budget +
   build-now-vs-defer verdict into `docs/source-evaluation.md`; cite sources.

## 10. Deliverable — `docs/source-evaluation.md`

Structure:
1. **Summary** — the recommended mix in two sentences + the headline B segment budget.
2. **Source-evaluation matrix** — sources × the 7 criteria (High/Med/Low + notes).
3. **Per-language coverage sub-table** — source × the 8 languages.
4. **Finalist probe results** — measured ⚙ numbers (capture rate, median clean speech, distinct
   channels) per finalist × language.
5. **Recommendation** — the mix (per-language where needed), with rationale tied to the gates/ranking.
6. **Workstream B segment budget** — target independent segments/language + min distinct
   channels/language, justified by probe numbers.
7. **Build-now-vs-defer verdict** — which fetchers B builds vs defers to Phase 1.
8. **Legal/attribution notes** + **honest caveats** (register trade-off, coverage gaps that remain).

## 11. Risks & mitigations

- **Probe network flakiness / geo-block recurs** → record failures as data (capture success rate is a
  ranking input, not a blocker); run controller-side with retries; a gated source that fails the probe
  is correctly demoted.
- **Corpus register pulls the map** → anchor is a *control*, flagged and analyzed separately, never
  blended into the primary natural-speech sample.
- **Scope creep into B** (probes becoming a full collection) → probes are capped at a handful of
  recordings/language and do **no** LLM verification or aggregation.
- **Legal ambiguity** (podcasts/broadcaster ToS) → deferred-but-informed: record the posture per
  source, exclude clearly-infringing, leave final legal sign-off to pre-publication.

## 12. Success criteria

- A **complete matrix** (all four families × seven criteria) + per-language coverage sub-table.
- **Measured** probe numbers for the 2–3 finalists across **all 8 languages**.
- A **defensible recommended mix** that, per the gates, plausibly yields independent verified clean
  segments across many channels for every seed language — with an honest account of any language whose
  coverage stays thin.
- A **concrete B segment budget** and a **build-now-vs-defer** verdict, both grounded in probe evidence.

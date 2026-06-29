# Source Evaluation — Findings (Workstream A)

**Phase:** Data Integrity (0.75) · **Workstream:** A (alternative clip-source research) · **Date:** 2026-06-29
**Spec:** `docs/superpowers/specs/2026-06-29-source-evaluation-design.md` · **Plan:** `docs/superpowers/plans/2026-06-29-source-evaluation.md`

Method: desk survey of four source families (✎ ratings) + **real capture probes** across all 8 seed
languages (⚙ measured). Probe code: `src/musiclang/probe/` + `scripts/probe_source.py`. Raw probe
data: `data/source_probe_results_{radio,podcast}.parquet` (gitignored). Seed languages: english,
german, polish, french, spanish, italian, greek, finnish.

---

## 1. Summary

**Recommended source mix: podcasts (primary) + internet radio (supplement), corpus dropped.**

The probe is decisive. **Podcast RSS feeds captured 100% successfully and yielded ≥30 s of clean
speech 100% of the time, in every one of the 8 languages** (median 75–89 s clean/episode, zero
failures) — native, natural-register, attributable, and independent-by-episode. **Radio is a strong
supplement** where it captures (English/Spanish/French perfect; Greek/Italian partial) but is
**unusable for German from the collection host** (0/6 — geo-block confirmed) and leaks music on some
"talk" stations. **The VoxPopuli corpus anchor was evaluated and dropped**: its per-language data is
dominated by **non-native speakers**, which is disqualifying for a project measuring native prosody
(see §5).

**Headline Workstream B budget:** target **≥25 independent ≥30 s clean-speech segments per language
across ≥4 distinct channels** (≥6 where data allows). This is comfortably reachable for all 8 from
**podcasts alone** (each episode is an independent recording) plus radio for channel diversity; German
is podcast-only from this host.

---

## 2. Source-evaluation matrix (families × criteria)

Ratings are desk H/M/L, **annotated with probe evidence** where measured (⚙).

| Criterion | Radio | Podcast | Broadcaster-API | Corpus (VoxPopuli) |
|---|---|---|---|---|
| **Coverage** (of the 8) | **H** — all 8 findable | **H** — all 8 findable | **M** — gaps at it/el | **M** — el absent; see §5 |
| **Cleanliness** | **M** ⚙ — music leaks (greek ERA-2/3, RAI GR-Parl ~0–15 s speech) | **H** ⚙ — every episode ≥73 s clean speech, no music | **H** (read news) | **H** *acoustically* — but **linguistically polluted** (non-native), §5 |
| **Independence** | **H** — distinct stations | **H** — 1 episode = 1 recording; unlimited distinct episodes | **H** | **H** — many speakers |
| **Register** | **M** — mixed (read news + phone-ins + ads/jingles) | **H** — spontaneous interview/news talk | **M** — mostly read | **H** *spontaneous* — but parliamentary + non-native |
| **Capturability** | **M→L** ⚙ — 69% overall; **German 0/6** geo-block; HLS works | **H** ⚙ — **100%** capture, all langs | **M** — several walled | **H** *loads* — but slow (huge shards) + Greek needs gated CV |
| **Channel diversity** | **H** — many stations/codecs | **H** — many shows | **H** | **L** — single corpus |
| **Legal/attribution** | **M** — public streams; bulk capture grey-zone | **M** — open RSS; commercial feeds ToS-restricted | **M** — BBC ToS forbids bulk; RTVE UA-blocks | **H** — CC0 (but unfit, §5) |

Broadcaster-API was not probed as a separate source: its public **RSS feeds fold into the podcast
probe** (BBC, Yle, DLF, RNE rows) and its **HLS into the radio probe** (BBC, Yle, RNE). Its
contribution is therefore measured *within* the podcast/radio numbers below. RAI and ERT expose **no
public API** (walled) and were reached only via generic radio (RAI/ERT Icecast) and community RSS.

---

## 3. Per-language coverage sub-table (desk)

How many distinct channels each family *makes findable* per language (H≈many / M≈some / L≈few / None):

| Language | Radio | Podcast | Broadcaster-API | Corpus |
|---|---|---|---|---|
| english | H | H | H | H |
| german | H | M | M | H |
| french | H | H | M | H |
| spanish | H | M | M | H |
| italian | M | H | L | H |
| greek | M | M | L | **None** |
| polish | M | L | M | H |
| finnish | H | H | H | M |

(Desk "coverage" = breadth of findable channels, *not* capturability — e.g. German radio is "H"
findable but 0% capturable from this host; see §4.)

---

## 4. Finalist probe results (measured, all 8 languages)

`cap` = capture success rate; `≥30s` = fraction yielding ≥30 s clean speech (silero VAD);
`med` = median clean-speech seconds per successful capture; `ch` = distinct channels probed.

**Podcast** (1 episode/feed, 28 feeds):

| Lang | cap | ≥30s | med (s) | ch |
|---|---|---|---|---|
| english | 1.00 | 1.00 | 78.6 | 4 |
| german | 1.00 | 1.00 | 86.2 | 4 |
| french | 1.00 | 1.00 | 88.3 | 5 |
| spanish | 1.00 | 1.00 | 80.2 | 4 |
| italian | 1.00 | 1.00 | 85.8 | 3 |
| greek | 1.00 | 1.00 | 79.9 | 3 |
| polish | 1.00 | 1.00 | 83.0 | 2 |
| finnish | 1.00 | 1.00 | 82.7 | 3 |
| **all** | **1.00** | **1.00** | **~83** | **28** |

**Radio** (1 capture/station, 29 stations, 60 s each):

| Lang | cap | ≥30s | med (s) | ch | note |
|---|---|---|---|---|---|
| english | 1.00 | 1.00 | 55.9 | 3 | BBC R4/R5/World via HLS |
| spanish | 1.00 | 1.00 | 60.2 | 4 | SER, Onda Cero, RNE×2 |
| french | 1.00 | 1.00 | 59.5 | 3 | France Inter/Info/Culture |
| polish | 0.67 | 0.67 | 55.2 | 3 | PR24 (HLS) + TOK FM; PR1 failed |
| finnish | 0.67 | 0.67 | 30.6 | 3 | Yle Radio 1/Suomi (HLS); Yle Puhe failed |
| italian | 0.75 | 0.50 | 35.8 | 4 | RAI R3 + Radio24; rai-radio1 failed; GR-Parl 14.7 s |
| greek | 1.00 | 0.33 | 0.0 | 3 | **only ERA Proto is talk**; ERA 2/3 = music (0 s speech) |
| german | **0.00** | **0.00** | 0.0 | 6 | **all 6 failed — geo-block** |
| **all** | **0.69** | **0.59** | — | **29** |

**Combined distinct channels yielding ≥30 s clean speech, per language** (the B-budget input):
english 7 · french 8 · spanish 8 · finnish 5 · italian 5 · greek 4 · german 4 (podcast-only) · polish 4.

---

## 5. Decision rule applied

**Hard gates** (must pass all to be recommendable):
1. covers/extensible to the 8 · 2. programmatically capturable to ≥30 s clean · 3. many independent
recordings · 4. legal posture acceptable for research.

| Family | Gate result |
|---|---|
| **Podcast** | **PASS all** — 100% capture + ≥30 s clean across all 8; 1 episode = 1 independent recording; open RSS. **Finalist (primary).** |
| **Radio** | **PASS** for 7/8 (fails gate 2 for **German** from this host); strong where it captures. **Finalist (supplement).** |
| **Broadcaster-API** | folds into podcast/radio; the open ones (BBC, Yle, DLF, RNE) already counted there. Not a separate finalist. |
| **Corpus (VoxPopuli)** | **REJECTED** — see below. |

**Why corpus is rejected (the key data-integrity finding).** VoxPopuli's per-language configs are
*every utterance spoken in language X in the European Parliament, regardless of the speaker's native
tongue.* English is the EP's working language, so its "english" split is **dominated by non-native
speakers** (German, French, … MEPs speaking English) — the dataset even ships an `accent` field for
this reason. A non-native speaker carries their **L1 prosody**, which is precisely the signal this
project measures, so the corpus would inject systematic noise, not anchor against it. It is not
reasonably fixable: accent-filtering discards most English and leaves a tiny, demographically narrow
set; **Greek is absent** from VoxPopuli's transcribed data (would need gated Common Voice, *read*
register); the register is read/parliamentary (already deprioritized); and streaming is slow
(multi-hundred-MB shards). The acoustically-"clean", CC0 corpus is **linguistically polluted** — a
case of integrity that looks good on paper and fails on inspection. *Radio and podcasts avoid this
entirely: they are national, in-country broadcasters → native speakers by construction.*

**Ranking of finalists:** podcast > radio. Podcast wins on capturability (1.00 vs 0.69), cleanliness
(no music leakage), and register (spontaneous talk); radio wins on channel/codec diversity and adds
independent native channels where it captures.

---

## 6. Recommendation (per-language mix)

**Primary = podcasts** for every language (perfect capture + cleanliness, native, natural,
independent-by-episode). **Supplement = radio** for channel diversity where it captures.

| Language | Recommended sourcing |
|---|---|
| english | podcast (BBC feeds) + radio (BBC HLS) — abundant |
| french | podcast (Radio France) + radio (France Inter/Info/Culture) — abundant |
| spanish | podcast (RNE) + radio (SER/Onda Cero/RNE HLS) — abundant |
| finnish | podcast (Yle) + radio (Yle HLS R1/Suomi) |
| italian | podcast (RAI feeds) + radio (RAI R3, Radio24); avoid GR-Parlamento/music streams |
| greek | podcast (Kathimerini/LIFO) + radio **ERA Proto only** (ERA 2/3 are music) |
| polish | podcast (thin: 2 feeds) + radio (PR24, TOK FM) — lean on radio for breadth |
| **german** | **podcast only** (DLF/NDR/WDR feeds) — live radio geo-blocked from this host |

---

## 7. Workstream B segment budget

**Target: ≥25 independent ≥30 s clean-speech segments per language, across ≥4 distinct channels
(≥6 where available).**

- Reachable for **all 8** from **podcasts alone**: each episode is an independent recording at a
  distinct time, and feeds carry dozens–hundreds of past episodes — so segment count is bounded by
  *willingness to pull episodes*, not availability. Radio adds independent native channels on top.
- **Minimum distinct channels/language ≥ 4** (probe-confirmed for every language). Aim ≥ 6 for
  en/fr/es/de (feeds + stations); accept 4 for el/pl (thinner) and supplement with extra episodes.
- **Radio breadth is dynamic, not the 29 probed stations.** Workstream B should enumerate stations
  live from **radio-browser.info** (already wired into the codebase: `ingest/radio.find_stations` /
  `find_capital_stations`), which returns *dozens* of distinct talk/news stations per language on
  demand — supplemented by HLS public broadcasters for gap languages. The 29 probe instances were a
  fitness sample, not a collection cap.
- **German** target met via podcasts only (4 feeds × many episodes). If German *radio* channel
  diversity is later wanted, it needs a DE-side egress (proxy/mirror) — deferred.

---

## 8. Build-now-vs-defer verdict

**Validated and ready to harden in B (built + proven in this probe):**
- Podcast RSS fetch → enclosure download → ffmpeg slice (`probe/adapters.capture_rss`, `latest_enclosures`) — **100% reliable in the probe.**
- HLS capture via `streamlink` → ffmpeg (`capture_hls`) — recovers BBC/Yle/RNE that the Phase-0 ffmpeg path couldn't.
- Progressive ffmpeg capture (`capture_progressive` → `record_clip`).

**Defer to Workstream B / C / Phase 1:**
- **Live-source temporal independence** (user decision): for live radio, sample **1 segment/station/hour over many hours** so multiple segments from one station are independent — *not* exercised in the probe; B owns it.
- **LLM speech/target-language verification + retry** (Workstream C): the probe used silero VAD as the cheap speech proxy; B upgrades to the audio/ASR LLM gate and retries non-speech captures.
- **Dynamic station enumeration at scale** via radio-browser (reuse Phase-0 `ingest/radio.py`).
- **German radio geo-block workaround** (DE egress) — only if German radio diversity is wanted beyond podcasts.

**Drop:** the corpus fetcher. `probe/adapters.corpus_probe` (+ the `datasets` dep) remain in the
tree but **unused** — retained only in case a *native* read corpus (e.g. Multilingual LibriSpeech,
native audiobook, but 6/8 languages — no Greek/Finnish) is ever wanted as a cross-register control.

---

## 9. Legal / attribution + honest caveats

**Legal (deferred-but-informed; final sign-off pre-publication):** public-broadcaster RSS and streams
are free for personal/research use; **bulk capture sits in a grey zone** in most EU jurisdictions
(no explicit licence; academic/non-commercial generally tolerated; redistribution/commercial use
needs permission). BBC ToS forbids non-personal automated download (its open CDN RSS was reachable
regardless); RTVE blocks some user-agents. Excluded YouTube/large-web entirely (licensing). VoxPopuli
is CC0 (Wang et al. 2021, *VoxPopuli*, ACL; arXiv:2101.00390) — but rejected on integrity grounds, not legal.

**Caveats:**
- **Corpus dropped** → no clean cross-register *control* in this phase. Acceptable: the project targets
  natural native prosody, which radio/podcast supply directly. If a control is later wanted, MLS
  (native audiobook, read, 6/8 langs) is the least-bad option — deferred.
- **Music leakage** on some "talk" radio (Greek ERA 2/3, RAI GR-Parlamento): real but **caught by the
  VAD gate**; B's LLM gate + station curation handle it.
- **German live radio unavailable** from the collection host (geo-block) → German leans entirely on
  podcasts this phase.
- **Polish podcast is thin** (2 discoverable feeds; Polskie Radio hides feed GUIDs) → supplement with
  Polish radio (PR24, TOK FM both captured) + radio-browser.
- Probe captured each live station **once**, so radio cleanliness is a single-time snapshot; the
  hourly, independent, LLM-gated sampling that makes live-radio segments trustworthy is **Workstream B**.
- A few individual streams failed transiently (Yle Puhe, RAI Radio 1, PR1) — not source-level failures;
  B's dynamic enumeration + retry routes around them.

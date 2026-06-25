# Capital-City Sampling + Language Guard — Findings

**Status:** exploration follow-up to Phase 0 (see `docs/phase0-findings.md`).
**Goal:** sample each seed language from radio stations in its **capital city** (to control
for regional/dialect variation), and drop foreign-language stations with a light LLM guard.

## Approach

- **Capital selection** (`ingest/radio.find_capital_stations`): geo-search within **60 km** of
  each capital (London, Berlin, Warsaw, Paris, Madrid, Rome, Athens, Helsinki), unioned across
  a set of speech tags (`talk, news, information, public radio, spoken word, current affairs`)
  because radio-browser's `tagList` is AND-semantics and `talk,news` alone is too restrictive
  near a capital. Falls back to nationwide when a capital is too thin (e.g. Athens, Helsinki).
- **Language guard** (`ingest/language_filter.is_in_language`): a light **OpenAI**
  (`gpt-4o-mini`, structured output) call that classifies whether a station broadcasts
  primarily in the target language, using the station **name + country + listed-language tags**.
  Fail-open (a transient API error keeps the station), cached per `(name, country, language)`.
- **Collector** (`scripts/collect_sample.py`): capital selection → guard → record up to N usable
  clips; if a language records 0 clips, fall back to nationwide talk/news, then to a broad
  (any-tag) search.

## Coverage (5 clips/lang × 60 s target)

All 8 seed languages obtained data. Per-language usable clips: english 4, german 5, polish 5,
french 5, spanish 5, greek 5, italian 3, finnish 2.

| Language | %V (mean) | vocalic nPVI (mean) |
|---|---|---|
| english | 42.7 | 80.3 |
| finnish | 37.3 | 76.6 |
| french  | 38.6 | 79.0 |
| german  | 36.3 | 72.0 |
| greek   | 41.7 | 80.5 |
| italian | 38.1 | 78.1 |
| polish  | 36.0 | 70.5 |
| spanish | 41.8 | 75.2 |

## Findings

1. **radio-browser language tags are unreliable — metadata-based language ID has a ceiling.**
   The BBC World Service foreign-language services are all tagged `english` in radio-browser:
   `BBC Afrique → "english,french"`, `BBC Arabic → "arabic,english"`, `BBC Somali →
   "english,somali"`. So no deterministic tag rule can exclude them, and an LLM judging from
   name+country+tags is also unreliable on them: the guard caught the unambiguous `BBC Arabic`
   but kept `BBC Afrique` (French) and `BBC Somali` (Somali) for English. **The robust fix is
   audio-based language detection** (transcribe a few seconds → language-ID) — a Phase-1 item.

2. **The guard's precision/recall is a genuine trade-off.**
   - *Name-only* (first cut) over-dropped generically-named local stations (9 false drops, e.g.
     `Alpha 98.9`, `RID 96.8`), thinning Italian to 1 clip.
   - *Country-aware + lenient* ("keep unless clearly a different language") eliminated those
     false positives (9 → 0; Italian recovered to 3) — but at the cost of letting the two
     mis-tagged BBC foreign services through (above). Metadata simply cannot separate
     "UK-based English station" from "UK-based foreign-language BBC service" reliably.

3. **German radio is largely geo-blocked from the collection host.**
   The capital and tagged-nationwide German stations are dominated by public broadcasters
   (WDR/BR/DLF) whose CDNs DNS-fail from here. A **broad (any-tag) search** surfaces working
   private stations (Kontrafunk, NIUS, Radio Hagen, detektor.fm) and recovered German (some
   music; the VAD speech gate + guard filter it). The collector now includes this broad
   last-resort fallback.

4. **Capital sampling did not materially change the typology picture.** nPVI stays clustered
   ~70–80 across languages; the alignment-free baseline still reproduces classical rhythm
   typology only weakly — consistent with `docs/phase0-findings.md`.

## Recommendations for Phase 1

- **Audio-based language ID** (e.g. Whisper/ASR or a langid model on a short snippet) instead of
  metadata — the only reliable way past the radio-browser tag problem.
- **Region-aware / mirror-resilient sourcing** for geo-blocked public broadcasters (or accept
  private-station substitutes per language).
- **Music/speech filtering** in `clean` so the broad fallback's music-heavy stations are handled
  robustly rather than relying on the VAD speech gate alone.

> Data (`data/`) is gitignored; notebook figures regenerate locally (notebooks are
> output-stripped via nbstripout — re-run to view).

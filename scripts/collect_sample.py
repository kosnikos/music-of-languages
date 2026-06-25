"""Collect a small real-radio sample and extract per-language prosody features.

Uses capital-city station selection (find_capital_stations) to prefer stations
near each language's capital before falling back nationwide, and an OpenAI
language guard (is_in_language) to drop obvious foreign-language stations.

Usage:
    python scripts/collect_sample.py [--clips-per-lang 5] [--clip-seconds 60] [--no-guard]

Requirements:
    - OPENAI_API_KEY set in the environment or in a .env file at the repo root
      (the guard is disabled gracefully if the key is absent / invalid).
    - ffmpeg must be on PATH (used by record_clip).

Writes:
    data/clips/<lang>/*.wav
    data/lang_features.parquet
    data/lang_cache.json   (persistent guard-verdict cache, gitignored)
"""

from __future__ import annotations

import argparse
import json

from musiclang.audio import load_audio, normalize_loudness
from musiclang.clean.vad import concat_speech, extract_speech, total_speech_seconds
from musiclang.config import DATA_DIR, SEED_LANGUAGES, TARGET_SAMPLE_RATE
from musiclang.features.aggregate import build_language_table
from musiclang.features.prosody_acoustic import ProsodyAcousticExtractor
from musiclang.ingest.radio import find_capital_stations, record_clip
from musiclang.ingest.language_filter import is_in_language


def collect(clips_per_lang: int, clip_seconds: int, no_guard: bool = False) -> None:
    # --- load env (reads .env from repo root → OPENAI_API_KEY) ---
    from dotenv import load_dotenv
    load_dotenv()

    # --- build OpenAI client once, gracefully ---
    if no_guard:
        oai_client = None
        print("[info] --no-guard: language guard disabled")
    else:
        try:
            from openai import OpenAI
            oai_client = OpenAI()  # raises if no OPENAI_API_KEY
        except Exception as exc:
            oai_client = None
            print(
                f"[warn] language guard disabled (no usable OpenAI client: {exc});"
                " recording all candidate stations"
            )

    # --- load persistent verdict cache ---
    cache_path = DATA_DIR / "lang_cache.json"
    if cache_path.exists():
        try:
            with cache_path.open(encoding="utf-8") as fh:
                cache: dict[str, bool] = json.load(fh)
        except Exception as exc:
            print(f"[warn] could not load cache ({exc}); starting fresh")
            cache = {}
    else:
        cache = {}

    def _save_cache() -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)

    extractor = ProsodyAcousticExtractor()
    per_language: dict[str, list[dict[str, float]]] = {}

    for key, spec in SEED_LANGUAGES.items():
        # Fetch a larger pool so the guard can drop some and we still get enough.
        pool_size = max(clips_per_lang * 2, clips_per_lang + 3)
        candidates = find_capital_stations(spec.radio_browser_lang, limit=pool_size)

        # Apply the language guard.
        if oai_client is not None:
            kept: list = []
            for station in candidates:
                if is_in_language(station.name, spec.name, client=oai_client, cache=cache):
                    kept.append(station)
                else:
                    print(f"[lang-skip] {key}: {station.name}")
            stations = kept
        else:
            stations = candidates

        # Record up to clips_per_lang usable clips.
        vectors: list[dict[str, float]] = []
        for i, station in enumerate(stations):
            if len(vectors) >= clips_per_lang:
                break
            out = DATA_DIR / "clips" / key / f"{i:02d}.wav"
            try:
                record_clip(station.url, out, duration_s=clip_seconds)
                signal = normalize_loudness(load_audio(out))
                segments = extract_speech(signal)
                if total_speech_seconds(segments) < 5.0:
                    continue
                speech = concat_speech(signal, segments)
                vectors.append(extractor.extract(speech, sr=TARGET_SAMPLE_RATE))
            except Exception as exc:  # prototype: log and continue
                print(f"[skip] {key} station {i}: {exc}")

        if vectors:
            per_language[key] = vectors
        print(f"{key}: {len(vectors)} usable clips")

        # Persist cache after each language.
        _save_cache()

    table = build_language_table(per_language)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(DATA_DIR / "lang_features.parquet")
    print(table)

    # Final cache save.
    _save_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect radio clips and extract prosody features per language."
    )
    parser.add_argument("--clips-per-lang", type=int, default=5)
    parser.add_argument("--clip-seconds", type=int, default=60)
    parser.add_argument(
        "--no-guard",
        action="store_true",
        help="Skip the OpenAI language guard and record all candidate stations.",
    )
    args = parser.parse_args()
    collect(args.clips_per_lang, args.clip_seconds, no_guard=args.no_guard)

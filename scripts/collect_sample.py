"""Collect a small real-radio sample and extract per-language prosody features.

Usage: python scripts/collect_sample.py --clips-per-lang 5 --clip-seconds 60
Writes data/clips/<lang>/*.wav and data/lang_features.parquet.
"""

from __future__ import annotations

import argparse

from musiclang.audio import load_audio, normalize_loudness
from musiclang.clean.vad import concat_speech, extract_speech, total_speech_seconds
from musiclang.config import DATA_DIR, SEED_LANGUAGES, TARGET_SAMPLE_RATE
from musiclang.features.aggregate import build_language_table
from musiclang.features.prosody_acoustic import ProsodyAcousticExtractor
from musiclang.ingest.radio import find_stations, record_clip


def collect(clips_per_lang: int, clip_seconds: int) -> None:
    extractor = ProsodyAcousticExtractor()
    per_language: dict[str, list[dict[str, float]]] = {}

    for key, spec in SEED_LANGUAGES.items():
        stations = find_stations(spec.radio_browser_lang, limit=clips_per_lang)
        vectors: list[dict[str, float]] = []
        for i, station in enumerate(stations[:clips_per_lang]):
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

    table = build_language_table(per_language)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(DATA_DIR / "lang_features.parquet")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-per-lang", type=int, default=5)
    parser.add_argument("--clip-seconds", type=int, default=60)
    args = parser.parse_args()
    collect(args.clips_per_lang, args.clip_seconds)

# scripts/collect_segments.py
"""Collect ONE verified independent 30s clean-speech segment per recording, per
language, from podcasts (primary) + radio (supplement). Workstream B+C.

  uv run python scripts/collect_segments.py --pilot           # 1 segment/language
  uv run python scripts/collect_segments.py --per-language 25 # full run

Requires ffmpeg + streamlink on PATH (see the plan Global Constraints) and
OPENAI_API_KEY in .env. `data/` is gitignored.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import soundfile as sf

from musiclang.clean.select import select_segment
from musiclang.config import DATA_DIR, SEED_LANGUAGES, TARGET_SAMPLE_RATE
from musiclang.ingest.manifest import (
    DROPS_COLUMNS, SEGMENTS_COLUMNS, drops_dataframe, segments_manifest_dataframe,
)
from musiclang.pipeline import clean_clip
from musiclang.probe import adapters
from musiclang.probe.core import RecordingRef
from musiclang.verify.verifier import verify_segment


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")[:48]


def process_recording(
    wav_path, *, language, source, channel_id, recording_ref, recorded_at,
    clean=clean_clip, select=select_segment, verify=verify_segment, sr=TARGET_SAMPLE_RATE,
):
    """Clean -> select 30s -> verify one recording. Returns ('kept', meta, samples) or ('drop', reason, detail)."""
    try:
        signal = clean(wav_path)
        if len(signal) == 0:
            return ("drop", "no-30s", "empty after VAD")
        seg = select(signal, sr)
        if seg is None:
            return ("drop", "no-30s", f"{len(signal)/sr:.1f}s clean < 30s")
        v = verify(seg, sr, language)
        if v.label != "target-speech":
            detail = v.detected_language or f"music={v.tagger_music:.2f}"
            return ("drop", v.label, f"{v.stage_decided}:{detail}")
        meta = {
            "segment_id": f"{language}_{_slug(channel_id)}_{_slug(recording_ref)}",
            "language": language, "source": source, "channel_id": channel_id,
            "recording_ref": recording_ref, "recorded_at": recorded_at,
            "clean_speech_s": float(len(signal) / sr), "path": "",
            "label": v.label, "confidence": v.confidence, "detected_language": v.detected_language,
            "transcript": v.transcript[:500], "tagger_speech": v.tagger_speech,
            "tagger_music": v.tagger_music, "stage_decided": v.stage_decided,
        }
        return ("kept", meta, seg)
    except Exception as exc:  # noqa: BLE001 — a bad recording is a drop, never a crash
        return ("drop", "error", str(exc)[:200])


def _channels_for(language: str, instances: pd.DataFrame, sources) -> list[dict]:
    """Per-language channel list from the Workstream-A instances (podcast feeds + radio)."""
    df = instances[(instances["language"] == language) & (instances["source"].isin(sources))]
    return df.to_dict("records")


def _recordings(channel: dict, per_channel: int):
    """Yield (recording_ref, capture_kind, capture_arg) for a channel."""
    if channel["kind"] == "rss_feed":
        for i, url in enumerate(adapters.latest_enclosures(channel["ref"], per_channel)):
            yield (f"{_slug(channel['channel_id'])}-ep{i}", "rss", url)
    else:  # hls | progressive  -> one capture per station
        yield (_slug(channel["channel_id"]), channel["kind"], channel["ref"])


def run(per_language=25, sources=("podcast", "radio"), pilot=False,
        instances_path=None, out_dir=None):
    instances_path = Path(instances_path or DATA_DIR / "source_instances.parquet")
    out_dir = Path(out_dir or DATA_DIR / "segments")
    work = DATA_DIR / "_seg_work"; work.mkdir(parents=True, exist_ok=True)
    instances = pd.read_parquet(instances_path)
    target = 1 if pilot else per_language
    per_channel = 1 if pilot else max(3, per_language // 2)

    seg_rows, drop_rows = [], []
    for language in SEED_LANGUAGES:
        kept = 0
        for ch in _channels_for(language, instances, sources):
            if kept >= target:
                break
            for rec_ref, kind, arg in _recordings(ch, per_channel):
                if kept >= target:
                    break
                ref = RecordingRef(ch["source"], language, ch["channel_id"], kind, arg)
                wav = work / f"{language}_{rec_ref}.wav"
                capture_fn = adapters.CAPTURE_DISPATCH.get(kind)
                if capture_fn is None or capture_fn(ref, wav) is None:
                    drop_rows.append({"language": language, "source": ch["source"],
                                      "channel_id": ch["channel_id"], "recording_ref": rec_ref,
                                      "reason": "capture-failed", "detail": kind})
                    continue
                recorded_at = datetime.now(timezone.utc).isoformat()
                status, a, b = process_recording(
                    wav, language=language, source=ch["source"], channel_id=ch["channel_id"],
                    recording_ref=rec_ref, recorded_at=recorded_at)
                if status == "kept":
                    seg_path = out_dir / language / f"{a['segment_id']}.wav"
                    seg_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(seg_path), b, TARGET_SAMPLE_RATE)
                    a["path"] = str(seg_path)
                    seg_rows.append(a)
                    kept += 1
                    print(f"[keep] {language} {a['channel_id']} {rec_ref} ({kept}/{target})")
                else:
                    drop_rows.append({"language": language, "source": ch["source"],
                                      "channel_id": ch["channel_id"], "recording_ref": rec_ref,
                                      "reason": a, "detail": b})
                    print(f"[drop:{a}] {language} {ch['channel_id']} {rec_ref}")
        print(f"=== {language}: {kept}/{target} verified ===")

    seg_df = segments_manifest_dataframe(seg_rows)
    drop_df = drops_dataframe(drop_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_pilot" if pilot else ""
    seg_df.to_parquet(DATA_DIR / f"segments_manifest{suffix}.parquet")
    drop_df.to_parquet(DATA_DIR / f"segments_drops{suffix}.parquet")
    print(f"\nWROTE {len(seg_df)} segments, {len(drop_df)} drops "
          f"({DATA_DIR / f'segments_manifest{suffix}.parquet'})")
    return seg_df, drop_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect verified independent 30s speech segments.")
    p.add_argument("--pilot", action="store_true", help="1 verified segment/language (review gate)")
    p.add_argument("--per-language", type=int, default=25)
    p.add_argument("--sources", type=str, default="podcast,radio")
    p.add_argument("--instances", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    run(per_language=args.per_language,
        sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
        pilot=args.pilot, instances_path=args.instances, out_dir=args.out)

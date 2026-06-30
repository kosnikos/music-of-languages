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
import concurrent.futures
import re
import threading
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pandas as pd
import soundfile as sf
from dotenv import load_dotenv

from musiclang.clean.select import select_segment
from musiclang.config import DATA_DIR, SEED_LANGUAGES, TARGET_SAMPLE_RATE
from musiclang.ingest.manifest import (
    DROPS_COLUMNS, SEGMENTS_COLUMNS, drops_dataframe, segments_manifest_dataframe,
)
from musiclang.pipeline import clean_clip
from musiclang.probe import adapters
from musiclang.probe.core import RecordingRef
from musiclang.verify.llm_judge import judge_transcript
from musiclang.verify.tagger import tag_speech_music
from musiclang.verify.verifier import verify_segment
from musiclang.verify.whisper_id import transcribe_language


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")[:48]


# All torch model inference — silero VAD inside clean_clip, AST inside tag_speech_music —
# is serialized through ONE lock. CPU torch forward() is NOT thread-safe across the worker
# pool: concurrent calls segfault. The slow work (captures + the Whisper/LLM API round-trips)
# stays fully parallel; the serialized torch parts are a small fraction of wall-clock.
_TORCH_LOCK = threading.Lock()


def _locked_clean(path):
    with _TORCH_LOCK:
        return clean_clip(path)


def _locked_tagger(signal, sr):
    with _TORCH_LOCK:
        return tag_speech_music(signal, sr)


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
        instances_path=None, out_dir=None, workers=16, warm=True):
    load_dotenv()

    instances_path = Path(instances_path or DATA_DIR / "source_instances.parquet")
    out_dir = Path(out_dir or DATA_DIR / "segments")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = DATA_DIR / "_seg_work"; work.mkdir(parents=True, exist_ok=True)
    instances = pd.read_parquet(instances_path)
    target = 1 if pilot else per_language
    episodes_per_feed = 1 if pilot else max(2, per_language // 4)

    # Build shared OpenAI client (all workers reuse it; avoids per-call construction overhead)
    client = None
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        pass

    def _verify(seg, sr, lang):
        return verify_segment(
            seg, sr, lang,
            tagger=_locked_tagger,
            whisper=partial(transcribe_language, client=client),
            llm_judge=partial(judge_transcript, client=client),
        )

    # Warm shared model singletons single-threaded before the thread pool
    if warm:
        try:
            from musiclang.verify.tagger import _load_ast; _load_ast()
        except Exception:
            pass
        try:
            from musiclang.clean.vad import _load_model; _load_model()
        except Exception:
            pass

    # Build flat worklist: (language, channel_dict, rec_ref, kind, arg)
    worklist = []
    for language in SEED_LANGUAGES:
        channels = _channels_for(language, instances, sources)
        lang_budget = int(target * 1.6) + 4 if not pilot else target * len(channels) + 4
        lang_count = 0
        for ch in channels:
            if lang_count >= lang_budget:
                break
            for rec_ref, kind, arg in _recordings(ch, episodes_per_feed):
                if lang_count >= lang_budget:
                    break
                worklist.append((language, ch, rec_ref, kind, arg))
                lang_count += 1

    # Worker job: capture + process; returns (language, source, channel_id, rec_ref, kind, status, a, b)
    def _job(item):
        language, ch, rec_ref, kind, arg = item
        ref = RecordingRef(ch["source"], language, ch["channel_id"], kind, arg)
        wav = work / f"{language}_{rec_ref}.wav"
        capture_fn = adapters.CAPTURE_DISPATCH.get(kind)
        if capture_fn is None or capture_fn(ref, wav) is None:
            return (language, ch["source"], ch["channel_id"], rec_ref, kind, "drop-capture", None, None)
        recorded_at = datetime.now(timezone.utc).isoformat()
        status, a, b = process_recording(
            wav, language=language, source=ch["source"], channel_id=ch["channel_id"],
            recording_ref=rec_ref, recorded_at=recorded_at, clean=_locked_clean, verify=_verify)
        return (language, ch["source"], ch["channel_id"], rec_ref, kind, status, a, b)

    # Aggregate results per-language; only main thread touches kept/drops/sf.write
    kept_by_lang: dict[str, list] = {lang: [] for lang in SEED_LANGUAGES}
    drop_rows = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, item): item for item in worklist}
        for fut in concurrent.futures.as_completed(futures):
            try:
                language, source, channel_id, rec_ref, kind, status, a, b = fut.result()
            except Exception as exc:
                # Defensive: job itself raised — treat as error drop (language unknown here)
                drop_rows.append({"language": "unknown", "source": "unknown",
                                   "channel_id": "unknown", "recording_ref": "unknown",
                                   "reason": "error", "detail": str(exc)[:200]})
                continue

            if status == "drop-capture":
                drop_rows.append({"language": language, "source": source,
                                   "channel_id": channel_id, "recording_ref": rec_ref,
                                   "reason": "capture-failed", "detail": kind})
                print(f"[drop:capture-failed] {language} {channel_id} {rec_ref}")
            elif status == "kept":
                if len(kept_by_lang[language]) < target:
                    seg_path = out_dir / language / f"{a['segment_id']}.wav"
                    seg_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(seg_path), b, TARGET_SAMPLE_RATE)
                    a["path"] = str(seg_path)
                    kept_by_lang[language].append(a)
                    n = len(kept_by_lang[language])
                    print(f"[keep] {language} {channel_id} {rec_ref} ({n}/{target})")
                # else: surplus — ignore (we oversampled)
            else:  # drop:reason
                drop_rows.append({"language": language, "source": source,
                                   "channel_id": channel_id, "recording_ref": rec_ref,
                                   "reason": a, "detail": b})
                print(f"[drop:{a}] {language} {channel_id} {rec_ref}")

    # Summary
    for language in SEED_LANGUAGES:
        n = len(kept_by_lang[language])
        print(f"=== {language}: {n}/{target} verified ===")

    seg_rows = [row for lang_rows in kept_by_lang.values() for row in lang_rows]
    seg_df = segments_manifest_dataframe(seg_rows)
    drop_df = drops_dataframe(drop_rows)
    suffix = "_pilot" if pilot else ""
    seg_df.to_parquet(out_dir / f"segments_manifest{suffix}.parquet")
    drop_df.to_parquet(out_dir / f"segments_drops{suffix}.parquet")
    print(f"\nWROTE {len(seg_df)} segments, {len(drop_df)} drops "
          f"({out_dir / f'segments_manifest{suffix}.parquet'})")
    return seg_df, drop_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect verified independent 30s speech segments.")
    p.add_argument("--pilot", action="store_true", help="1 verified segment/language (review gate)")
    p.add_argument("--per-language", type=int, default=25)
    p.add_argument("--sources", type=str, default="podcast,radio")
    p.add_argument("--instances", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    run(per_language=args.per_language,
        sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
        pilot=args.pilot, instances_path=args.instances, out_dir=args.out,
        workers=args.workers)

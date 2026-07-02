"""Probe candidate clip sources: capture a few recordings/source/language and
measure capturability + cleanliness + independence. Workstream A (throwaway).

Usage:
    uv run python scripts/probe_source.py [--n-per-language 4]
        [--sources radio,podcast,corpus] [--instances data/source_instances.parquet]
        [--out data/source_probe_results.parquet]

Requires ffmpeg + streamlink reachable on PATH for real runs (see the plan's
Task 6 for the Windows PATH handling). `data/` is gitignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from musiclang.audio import load_audio, normalize_loudness
from musiclang.clean.vad import extract_speech
from musiclang.config import DATA_DIR, SEED_LANGUAGES, TARGET_SAMPLE_RATE
from musiclang.probe import adapters
from musiclang.probe.core import ProbeResult, RecordingRef, measure_cleanliness


def expand_instances(rows, k: int, *, enclosure_fn=None) -> list[RecordingRef]:
    """Instance rows -> RecordingRefs; expand each 'rss_feed' into up to k episodes."""
    enclosure_fn = enclosure_fn or adapters.latest_enclosures
    refs: list[RecordingRef] = []
    for row in rows:
        if row["kind"] == "rss_feed":
            for url in enclosure_fn(row["ref"], k):
                refs.append(RecordingRef(
                    row["source"], row["language"], row["channel_id"], "rss", url
                ))
        else:
            refs.append(RecordingRef(
                row["source"], row["language"], row["channel_id"], row["kind"], row["ref"]
            ))
    return refs


def probe_ref(
    ref: RecordingRef, work_dir, *, capture_dispatch=None,
    load=load_audio, normalize=normalize_loudness, speech_fn=extract_speech,
) -> ProbeResult:
    """Capture one ref, measure clean-speech seconds. Never raises — errors -> ProbeResult."""
    capture_dispatch = capture_dispatch if capture_dispatch is not None else adapters.CAPTURE_DISPATCH
    fn = capture_dispatch.get(ref.kind)
    safe_id = ref.channel_id.replace("/", "_")[:40]
    out = Path(work_dir) / f"{ref.source}_{ref.language}_{safe_id}_{ref.kind}.wav"
    try:
        path = fn(ref, out) if fn is not None else None
        if path is None:
            return ProbeResult(ref.source, ref.language, ref.channel_id, ref.kind,
                               False, 0.0, False, "capture-failed")
        signal = normalize(load(path))
        clean_s, meets = measure_cleanliness(signal, TARGET_SAMPLE_RATE, speech_fn=speech_fn)
        return ProbeResult(ref.source, ref.language, ref.channel_id, ref.kind,
                           True, float(clean_s), bool(meets))
    except Exception as exc:  # noqa: BLE001 — a probe failure is data, not a crash
        return ProbeResult(ref.source, ref.language, ref.channel_id, ref.kind,
                           False, 0.0, False, str(exc)[:200])


def summarize(results: list[ProbeResult]) -> pd.DataFrame:
    """Per (source, language): N, capture rate, 30s-meet rate, median clean s, distinct channels."""
    if not results:
        return pd.DataFrame(
            columns=["source", "language", "n", "capture_rate",
                     "meets_30s_rate", "median_clean_s", "distinct_channels"]
        )
    df = pd.DataFrame([r.__dict__ for r in results])
    grouped = df.groupby(["source", "language"])
    out = grouped.agg(
        n=("capturable", "size"),
        capture_rate=("capturable", "mean"),
        meets_30s_rate=("meets_30s", "mean"),
        median_clean_s=("clean_speech_s", "median"),
        distinct_channels=("channel_id", "nunique"),
    ).reset_index()
    return out


def run(
    n_per_language: int = 4,
    sources: tuple[str, ...] = ("radio", "podcast", "corpus"),
    instances_path: str | Path = None,
    out_path: str | Path = None,
) -> pd.DataFrame:
    instances_path = Path(instances_path or DATA_DIR / "source_instances.parquet")
    out_path = Path(out_path or DATA_DIR / "source_probe_results.parquet")
    work_dir = DATA_DIR / "probe_clips"
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[ProbeResult] = []

    # --- radio + podcast: from the research instances file ---
    if instances_path.exists():
        inst = pd.read_parquet(instances_path)
        inst = inst[inst["source"].isin([s for s in sources if s in ("radio", "podcast")])]
        refs = expand_instances(inst.to_dict("records"), k=n_per_language)
        for ref in refs:
            res = probe_ref(ref, work_dir)
            results.append(res)
            print(f"[{res.source}/{res.language}] {res.channel_id}: "
                  f"{'ok' if res.capturable else res.error} clean={res.clean_speech_s:.1f}s")

    # --- corpus: VoxPopuli (7 langs) + Common Voice (Greek) ---
    if "corpus" in sources:
        for lang in SEED_LANGUAGES:
            try:
                pairs = adapters.corpus_probe(lang, n_per_language, work_dir / "corpus" / lang)
            except Exception as exc:  # noqa: BLE001
                print(f"[corpus/{lang}] load failed: {exc}")
                continue
            for ref, _wav in pairs:
                res = probe_ref(ref, work_dir)
                results.append(res)
                print(f"[corpus/{lang}] {res.channel_id}: clean={res.clean_speech_s:.1f}s")

    df = pd.DataFrame([r.__dict__ for r in results])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    summary = summarize(results)
    print("\n=== per-source/language summary ===")
    print(summary.to_string(index=False))
    summary.to_parquet(out_path.with_name("source_probe_summary.parquet"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe candidate clip sources.")
    parser.add_argument("--n-per-language", type=int, default=4)
    parser.add_argument("--sources", type=str, default="radio,podcast,corpus")
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    run(
        n_per_language=args.n_per_language,
        sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
        instances_path=args.instances,
        out_path=args.out,
    )

"""Embed the 178 verified segments once with XLS-R (layers {12,16,last}) and cache.

Controller background job: single-process sequential (concurrent CPU torch forward
passes segfaulted in Workstream B), resumable (skips already-cached segment_ids).
Run: uv run python -u scripts/embed_segments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from musiclang.audio import load_audio
from musiclang.config import TARGET_SAMPLE_RATE
from musiclang.features.ssl_embedding import SSLEmbeddingExtractor

DATA_DIR = Path("data/segments")
MANIFEST = DATA_DIR / "segments_manifest_final.parquet"
OUT_DIR = Path("data")
LAYERS = (12, 16, -1)


def _layer_filename(layer: int, out_dir: Path = OUT_DIR) -> str:
    tag = "last" if layer == -1 else str(layer)
    return str(out_dir / f"segment_embeddings_xlsr_l{tag}.parquet")


def _already_done(layers, out_dir) -> set[str]:
    done: set[str] | None = None
    for ly in layers:
        p = Path(_layer_filename(ly, out_dir))
        ids = set(pd.read_parquet(p).index) if p.exists() else set()
        done = ids if done is None else (done & ids)  # done only if in ALL layer files
    return done or set()


def embed_segments(manifest, extractor, layers=LAYERS, out_dir=OUT_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _already_done(layers, out_dir)
    todo = manifest[~manifest["segment_id"].isin(done)]
    per_layer: dict[int, list[dict]] = {ly: [] for ly in layers}
    for n, (_, row) in enumerate(todo.iterrows(), 1):
        signal = load_audio(row["path"], sr=TARGET_SAMPLE_RATE)
        vecs = extractor.extract_layers(signal, sr=TARGET_SAMPLE_RATE, layers=layers)
        for ly in layers:
            per_layer[ly].append({"segment_id": row["segment_id"], **vecs[ly]})
        print(f"[{n}/{len(todo)}] {row['segment_id']}", flush=True)
    for ly in layers:
        if not per_layer[ly]:
            continue
        new = pd.DataFrame(per_layer[ly]).set_index("segment_id")
        p = Path(_layer_filename(ly, out_dir))
        if p.exists():
            new = pd.concat([pd.read_parquet(p), new])
            new = new[~new.index.duplicated(keep="first")]
        new.to_parquet(p)
        print(f"wrote {p} ({len(new)} rows)", flush=True)


def main() -> int:
    manifest = pd.read_parquet(MANIFEST)
    extractor = SSLEmbeddingExtractor(model_id="facebook/wav2vec2-xls-r-300m", pooling="mean")
    embed_segments(manifest, extractor)
    return 0


if __name__ == "__main__":
    sys.exit(main())

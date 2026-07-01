"""Embed the 178 verified segments once with XLS-R (layers {12,16,last}) and cache.

Controller background job: single-process sequential (concurrent CPU torch forward
passes segfaulted in Workstream B), resumable (skips already-cached segment_ids).
Run: uv run python -u scripts/embed_segments.py
"""
from __future__ import annotations

import os
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
    """Return segment_ids present in every layer's cache file (else re-embed)."""
    done: set[str] | None = None
    for ly in layers:
        p = Path(_layer_filename(ly, out_dir))
        ids = set(pd.read_parquet(p).index) if p.exists() else set()
        done = ids if done is None else (done & ids)  # done only if in ALL layer files
    return done or set()


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def embed_segments(manifest, extractor, layers=LAYERS, out_dir=OUT_DIR, flush_every: int = 1) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # seed from existing caches so prior rows are never lost on rewrite
    frames: dict[int, pd.DataFrame | None] = {}
    for ly in layers:
        p = Path(_layer_filename(ly, out_dir))
        frames[ly] = pd.read_parquet(p) if p.exists() else None
    done = _already_done(layers, out_dir)
    todo = manifest[~manifest["segment_id"].isin(done)]
    pending: dict[int, list[dict]] = {ly: [] for ly in layers}

    def flush() -> None:
        for ly in layers:
            if not pending[ly]:
                continue
            new = pd.DataFrame(pending[ly]).set_index("segment_id")
            frames[ly] = new if frames[ly] is None else pd.concat([frames[ly], new])
            frames[ly] = frames[ly][~frames[ly].index.duplicated(keep="first")]
            _atomic_to_parquet(frames[ly], Path(_layer_filename(ly, out_dir)))
            pending[ly] = []

    for n, (_, row) in enumerate(todo.iterrows(), 1):
        signal = load_audio(row["path"], sr=TARGET_SAMPLE_RATE)
        vecs = extractor.extract_layers(signal, sr=TARGET_SAMPLE_RATE, layers=layers)
        for ly in layers:
            pending[ly].append({"segment_id": row["segment_id"], **vecs[ly]})
        print(f"[{n}/{len(todo)}] {row['segment_id']}", flush=True)
        if n % flush_every == 0:
            flush()
    flush()  # final flush for any remainder


def main() -> int:
    manifest = pd.read_parquet(MANIFEST)
    extractor = SSLEmbeddingExtractor(model_id="facebook/wav2vec2-xls-r-300m", pooling="mean")
    embed_segments(manifest, extractor)
    return 0


if __name__ == "__main__":
    sys.exit(main())

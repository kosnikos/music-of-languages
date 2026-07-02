"""Prosody (16-scalar) feature table + shared provenance for the 178 segments.

Cheap CPU pass; run synchronously. Run: uv run python scripts/build_segment_prosody.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from musiclang import pipeline
from musiclang.features.prosody_acoustic import ProsodyAcousticExtractor

MANIFEST = Path("data/segments/segments_manifest_final.parquet")
OUT_DIR = Path("data")


def build(manifest, extractor, out_dir=OUT_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_df, feat_df = pipeline.build_segment_features_direct(manifest, extractor)
    feat_df.to_parquet(out_dir / "segment_features_prosody.parquet")
    prov_df.to_parquet(out_dir / "segment_provenance.parquet")
    print(f"prosody: {feat_df.shape}, provenance: {prov_df.shape}", flush=True)


def main() -> int:
    manifest = pd.read_parquet(MANIFEST)
    build(manifest, ProsodyAcousticExtractor())
    return 0


if __name__ == "__main__":
    sys.exit(main())

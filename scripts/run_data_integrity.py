"""Data-integrity analysis driver: the before/after report + figures.

Integration capstone of the Data Integrity D+E cycle (Tasks 1-8): assembles a
single results dict from the outlier detectors (Task 6), robustness/stability
helpers (Task 7), and proximity-agreement/family-tree metrics (Tasks 4-5) —
comparing the full 178-segment set against an outlier-excluded set, and the
SSL embedding across layers 12/16/last, to check whether language separation
survives scrutiny or is instead a station/channel confound.

`assemble_report` is pure (synthetic-frame unit-testable); `load_inputs`,
`make_figures`, `write_results`, and `main` are thin I/O wrappers.

Run: uv run python scripts/run_data_integrity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from musiclang.proximity.distance import standardize, distance_matrix, linkage_matrix, mds_2d
from musiclang.validation.outliers import CentroidMADDetector, IsolationForestDetector, detect_language_outliers
from musiclang.validation.robustness import proximity_pipeline, leave_one_station_out, bootstrap_metric_ci
from musiclang.validation.proximity_agreement import confound_report, class_silhouette, within_between_separation
from musiclang.validation.family_tree import reference_distance_matrix, mantel_test, LINEAGE
from musiclang.validation.typology import RHYTHM_CLASS

DATA_DIR = Path("data")
PROSODY_PATH = DATA_DIR / "segment_features_prosody.parquet"
PROVENANCE_PATH = DATA_DIR / "segment_provenance.parquet"
EMBEDDING_PATHS = {
    12: DATA_DIR / "segment_embeddings_xlsr_l12.parquet",
    16: DATA_DIR / "segment_embeddings_xlsr_l16.parquet",
    "last": DATA_DIR / "segment_embeddings_xlsr_llast.parquet",
}
RESULTS_PATH = DATA_DIR / "data_integrity_results.json"
OUTLIERS_PATH = DATA_DIR / "segment_outliers.parquet"
FIG_DIR = Path("docs/figures/data-integrity")

# Phase-0.5 reference confound gaps (segment-level, SSL embeddings) — the "before" in
# this task's before/after comparison. See docs/data-integrity design spec.
PHASE05 = {"language_gap": 0.0125, "station_gap": 0.0491}


def _l2_normalize(emb_df: pd.DataFrame) -> pd.DataFrame:
    """Unit-normalize the emb_* columns of `emb_df` (rows), dropping non-embedding columns."""
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    x = emb_df[emb_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb_df[emb_cols].copy()
    unit[emb_cols] = x / norms
    return unit


def _segment_confound(seg_dist: pd.DataFrame, prov_df: pd.DataFrame, phase05: dict) -> dict:
    lang = prov_df["language"].to_dict()
    stat = prov_df["channel_id"].to_dict()
    rep = confound_report(seg_dist, lang, stat)
    rep["phase05"] = phase05
    return rep


MANTEL_PERMUTATIONS = 1_000  # mantel_test defaults to 10_000; that's ~4x too slow to call 4x per report


def _metrics(dist: pd.DataFrame, langs: list[str]) -> dict:
    """Headline language-level proximity-agreement metrics for one segment subset."""
    r, p = mantel_test(dist, reference_distance_matrix(langs), permutations=MANTEL_PERMUTATIONS)
    return {
        "rhythm_silhouette": class_silhouette(dist, RHYTHM_CLASS),
        "within_between": within_between_separation(dist, {l: l for l in dist.index}),
        "mantel_r": r,
        "mantel_p": p,
    }


def _delta(full: dict, excluded: dict) -> dict:
    """excluded - full, recursively, for the nested `_metrics` scalars."""
    if isinstance(full, dict):
        return {k: _delta(full[k], excluded[k]) for k in full}
    return excluded - full


def _rhythm_silhouette_fn(dist: pd.DataFrame) -> float:
    return class_silhouette(dist, RHYTHM_CLASS)


def compute_stability(feat_df: pd.DataFrame, prov_df: pd.DataFrame, method: str) -> dict:
    """leave-one-station-out spread + bootstrap CI (n_boot=1000) on the headline rhythm silhouette.

    Both helpers only expose a LANGUAGE-level proximity matrix to `metric_fn` (via
    `proximity_pipeline`), so the segment-level confound `station_gap` (which needs
    per-segment station identity) isn't reachable through this pair of functions —
    see `_segment_confound`/`confound_report` for that direct measurement instead.

    NOT called from `assemble_report`: with n_boot=1000 this is too slow for the
    synthetic-frame smoke test, so `main()` calls it directly on the real data
    after `assemble_report` returns, and merges the result in.
    """
    loso = leave_one_station_out(feat_df, prov_df, _rhythm_silhouette_fn, method=method)
    boot = bootstrap_metric_ci(feat_df, prov_df, _rhythm_silhouette_fn, method=method, n_boot=1000, seed=0)
    return {
        "leave_one_station_out": {
            "min": float(loso["metric"].min()),
            "median": float(loso["metric"].median()),
            "max": float(loso["metric"].max()),
        },
        "bootstrap_ci": boot,
    }


def _method_report(feat_df: pd.DataFrame, prov_df: pd.DataFrame, method: str, phase05: dict,
                    outlier_ids: set[str]) -> dict:
    if method == "prosody":
        seg_dist = distance_matrix(standardize(feat_df), metric="euclidean")
    else:
        seg_dist = distance_matrix(_l2_normalize(feat_df), metric="cosine")
    confound = _segment_confound(seg_dist, prov_df, phase05)

    langs = sorted(prov_df["language"].unique())
    full_dist = proximity_pipeline(feat_df, prov_df, method=method)
    excluded_dist = proximity_pipeline(feat_df, prov_df, method=method, exclude=outlier_ids)
    full_metrics = _metrics(full_dist, langs)
    excluded_metrics = _metrics(excluded_dist, langs)

    return {
        "confound": confound,
        "metrics": {
            "full": full_metrics,
            "excluded": excluded_metrics,
            "delta": _delta(full_metrics, excluded_metrics),
        },
    }


def _detect_outliers(prosody_df: pd.DataFrame, emb16_df: pd.DataFrame, prov_df: pd.DataFrame) -> pd.DataFrame:
    """Per Design point 2: CentroidMAD-prosody, CentroidMAD-ssl@cosine, IsolationForest-prosody."""
    labels = prov_df["language"].to_dict()
    frames = [
        detect_language_outliers(prosody_df, labels, CentroidMADDetector(), space="prosody"),
        detect_language_outliers(emb16_df, labels, CentroidMADDetector(metric="cosine"), space="ssl"),
        detect_language_outliers(prosody_df, labels, IsolationForestDetector(), space="prosody"),
    ]
    return pd.concat(frames, ignore_index=True)


def assemble_report(prosody_df: pd.DataFrame, emb16_df: pd.DataFrame, prov_df: pd.DataFrame, phase05: dict) -> dict:
    """Pure: build the full before/after report from in-memory frames (no I/O)."""
    outliers_df = _detect_outliers(prosody_df, emb16_df, prov_df)
    outlier_ids = set(outliers_df.loc[outliers_df["is_outlier"], "segment_id"])
    outlier_counts = {
        detector: int(grp["is_outlier"].sum()) for detector, grp in outliers_df.groupby("detector")
    }
    outlier_counts["union"] = len(outlier_ids)

    prosody_rep = _method_report(prosody_df, prov_df, "prosody", phase05, outlier_ids)
    ssl_rep = _method_report(emb16_df, prov_df, "ssl", phase05, outlier_ids)
    prosody_rep["outlier_counts"] = outlier_counts
    ssl_rep["outlier_counts"] = outlier_counts

    return {
        "prosody": prosody_rep,
        "ssl": ssl_rep,
        # Layer-16 entry is derived straight from `ssl_rep` above (no recomputation);
        # layers 12/last are filled in by `main()`, which alone has those frames.
        "layer_sensitivity": {
            "16": {
                "confound": ssl_rep["confound"],
                "rhythm_silhouette": ssl_rep["metrics"]["full"]["rhythm_silhouette"],
            },
        },
        "outliers": outliers_df,
    }


def _layer_entry(emb_df: pd.DataFrame, prov_df: pd.DataFrame, phase05: dict) -> dict:
    """Same confound + rhythm-silhouette pair as the ssl entry, for one SSL layer."""
    seg_dist = distance_matrix(_l2_normalize(emb_df), metric="cosine")
    lang_dist = proximity_pipeline(emb_df, prov_df, method="ssl")
    return {
        "confound": _segment_confound(seg_dist, prov_df, phase05),
        "rhythm_silhouette": class_silhouette(lang_dist, RHYTHM_CLASS),
    }


def load_inputs() -> tuple[pd.DataFrame, dict[int | str, pd.DataFrame], pd.DataFrame]:
    """Read the prosody/provenance/embedding parquets built by earlier tasks."""
    required = [PROSODY_PATH, PROVENANCE_PATH, *EMBEDDING_PATHS.values()]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data-integrity inputs: " + ", ".join(str(p) for p in missing) + ". "
            "Run scripts/build_segment_prosody.py and Task 3's background embedding job "
            "(scripts/embed_segments.py) first."
        )
    prosody_df = pd.read_parquet(PROSODY_PATH)
    prov_df = pd.read_parquet(PROVENANCE_PATH)
    emb = {layer: pd.read_parquet(path) for layer, path in EMBEDDING_PATHS.items()}
    return prosody_df, emb, prov_df


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to native Python (JSON can't serialize numpy)."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items() if k != "outliers"}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_results(rep: dict, results_path: Path = RESULTS_PATH, outliers_path: Path = OUTLIERS_PATH) -> None:
    outliers_df = rep.get("outliers")
    if outliers_df is not None:
        Path(outliers_path).parent.mkdir(parents=True, exist_ok=True)
        outliers_df.to_parquet(outliers_path)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(_to_native(rep), f, indent=2)


def make_figures(prosody_df: pd.DataFrame, emb16_df: pd.DataFrame, prov_df: pd.DataFrame,
                  rep: dict, out_dir: Path = FIG_DIR) -> None:
    """Dendrogram + language MDS (both methods) and segment MDS (SSL, by language/channel).

    Not smoke-tested (plotting only); runs in the controller's real `main()`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rhythm_colors = {"stress": "tab:red", "syllable": "tab:blue", "intermediate": "tab:green"}

    for method, feat_df in (("prosody", prosody_df), ("ssl", emb16_df)):
        lang_dist = proximity_pipeline(feat_df, prov_df, method=method)
        langs = list(lang_dist.index)
        fig, (ax_dendro, ax_mds) = plt.subplots(1, 2, figsize=(12, 5))
        dendrogram(linkage_matrix(lang_dist), labels=langs, ax=ax_dendro)
        ax_dendro.set_title(f"{method}: language dendrogram")

        coords = mds_2d(lang_dist)
        for lang in langs:
            cls = RHYTHM_CLASS.get(lang, "intermediate")
            ax_mds.scatter(coords.loc[lang, "mds_x"], coords.loc[lang, "mds_y"],
                            color=rhythm_colors.get(cls, "gray"), s=80)
            clade = "/".join(LINEAGE.get(lang, [lang]))
            ax_mds.annotate(f"{lang}\n{clade}", (coords.loc[lang, "mds_x"], coords.loc[lang, "mds_y"]), fontsize=7)
        ax_mds.set_title(f"{method}: language MDS (color = rhythm class)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_dendrogram_mds.png", dpi=150)
        plt.close(fig)

    seg_dist = distance_matrix(_l2_normalize(emb16_df), metric="cosine")
    seg_coords = mds_2d(seg_dist)
    fig, (ax_lang, ax_chan) = plt.subplots(1, 2, figsize=(12, 5))
    for lang, grp in prov_df.groupby("language"):
        ids = [i for i in grp.index if i in seg_coords.index]
        ax_lang.scatter(seg_coords.loc[ids, "mds_x"], seg_coords.loc[ids, "mds_y"], label=lang, s=20)
    ax_lang.set_title("segment MDS (SSL) — colored by language")
    ax_lang.legend(fontsize=6)
    for chan, grp in prov_df.groupby("channel_id"):
        ids = [i for i in grp.index if i in seg_coords.index]
        ax_chan.scatter(seg_coords.loc[ids, "mds_x"], seg_coords.loc[ids, "mds_y"], label=chan, s=20)
    ax_chan.set_title("segment MDS (SSL) — colored by channel_id")
    fig.tight_layout()
    fig.savefig(out_dir / "segment_mds_ssl.png", dpi=150)
    plt.close(fig)


def main() -> int:
    prosody_df, emb, prov_df = load_inputs()
    rep = assemble_report(prosody_df, emb[16], prov_df, phase05=PHASE05)
    rep["layer_sensitivity"]["12"] = _layer_entry(emb[12], prov_df, PHASE05)
    rep["layer_sensitivity"]["last"] = _layer_entry(emb["last"], prov_df, PHASE05)
    rep["prosody"]["stability"] = compute_stability(prosody_df, prov_df, "prosody")
    rep["ssl"]["stability"] = compute_stability(emb[16], prov_df, "ssl")
    write_results(rep)
    make_figures(prosody_df, emb[16], prov_df, rep)
    print(f"wrote {RESULTS_PATH}, {OUTLIERS_PATH}, and figures under {FIG_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

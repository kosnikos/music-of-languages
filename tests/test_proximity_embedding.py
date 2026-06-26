import numpy as np
import pandas as pd

from musiclang.proximity.embedding import language_centroids


def _df():
    # english: 2 segments from clip A, 1 from clip B; french: 1 segment
    return pd.DataFrame(
        {
            "language": ["english", "english", "english", "french"],
            "clip_id":  ["A", "A", "B", "C"],
            "emb_000":  [1.0, 1.0, 0.0, 0.0],
            "emb_001":  [0.0, 0.0, 1.0, 1.0],
        },
        index=["A_w000", "A_w001", "B_w000", "C_w000"],
    )


def test_unit_normalized_and_indexed_by_language():
    cent = language_centroids(_df())
    assert list(cent.index) == ["english", "french"]
    assert list(cent.columns) == ["emb_000", "emb_001"]
    # french has one unit vector (0,1)
    np.testing.assert_allclose(cent.loc["french"].to_numpy(), [0.0, 1.0])


def test_recording_vs_flat_weighting_differ():
    flat = language_centroids(_df(), weighting="flat")
    rec = language_centroids(_df(), weighting="recording")
    # flat: mean of [(1,0),(1,0),(0,1)] = (0.667, 0.333)
    np.testing.assert_allclose(flat.loc["english"].to_numpy(), [2 / 3, 1 / 3])
    # recording: clipA centroid (1,0), clipB (0,1) -> mean (0.5, 0.5)
    np.testing.assert_allclose(rec.loc["english"].to_numpy(), [0.5, 0.5])

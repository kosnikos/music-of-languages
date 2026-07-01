import numpy as np
import pandas as pd

from musiclang.proximity.embedding import language_centroids


def test_channel_weighting_balances_uneven_channels():
    rows = [{"language": "english", "channel_id": "X", "clip_id": f"x{i}",
             "emb_000": 1.0, "emb_001": 0.0} for i in range(3)]
    rows.append({"language": "english", "channel_id": "Y", "clip_id": "y0",
                 "emb_000": 0.0, "emb_001": 1.0})
    emb_df = pd.DataFrame(rows)
    cent = language_centroids(emb_df, weighting="channel")
    # channels X and Y weigh equally -> unit([1,0]) & unit([0,1]) averaged = [0.5, 0.5]
    assert np.isclose(cent.loc["english", "emb_000"], 0.5)
    assert np.isclose(cent.loc["english", "emb_001"], 0.5)
    # contrast: flat weighting would give [0.75, 0.25]
    flat = language_centroids(emb_df, weighting="flat")
    assert np.isclose(flat.loc["english", "emb_000"], 0.75)

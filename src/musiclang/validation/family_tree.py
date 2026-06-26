"""Genealogical reference distances + a Mantel test for proximity agreement.

The reference is a transparent tree-path distance over the seed languages'
classification (shared-lineage depth). Classification follows Glottolog
(Hammarström, Forkel, Haspelmath & Bank, https://glottolog.org). A richer
alternative is ASJP lexical distance (Wichmann et al., https://asjp.clld.org) —
noted for a later pass.

Mantel test: Mantel (1967), Cancer Research 27:209-220. The matrix-permutation
p-value compares the observed correlation against correlations under random
relabelings of one matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Lineage from broad family -> language (Glottolog classification).
LINEAGE: dict[str, list[str]] = {
    "english": ["IndoEuropean", "Germanic", "WestGermanic", "English"],
    "german":  ["IndoEuropean", "Germanic", "WestGermanic", "German"],
    "polish":  ["IndoEuropean", "BaltoSlavic", "Slavic", "WestSlavic", "Polish"],
    "french":  ["IndoEuropean", "Italic", "Romance", "WesternRomance", "French"],
    "spanish": ["IndoEuropean", "Italic", "Romance", "WesternRomance", "Spanish"],
    "italian": ["IndoEuropean", "Italic", "Romance", "Italian"],
    "greek":   ["IndoEuropean", "Hellenic", "Greek"],
    "finnish": ["Uralic", "Finnic", "Finnish"],
}


def _tree_distance(a: str, b: str) -> int:
    la, lb = LINEAGE[a], LINEAGE[b]
    shared = 0
    for x, y in zip(la, lb):
        if x == y:
            shared += 1
        else:
            break
    return (len(la) - shared) + (len(lb) - shared)


def reference_distance_matrix(languages: list[str]) -> pd.DataFrame:
    """Square genealogical tree-path distance matrix for `languages`."""
    mat = [[float(_tree_distance(a, b)) for b in languages] for a in languages]
    return pd.DataFrame(mat, index=languages, columns=languages)


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if method == "spearman":
        a, b = rankdata(a), rankdata(b)
    elif method != "pearson":
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    return float(np.corrcoef(a, b)[0, 1])


def mantel_test(
    dist_a: pd.DataFrame,
    dist_b: pd.DataFrame,
    method: str = "pearson",
    permutations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Correlation between two distance matrices + a one-sided permutation p-value."""
    order = list(dist_a.index)
    A = dist_a.loc[order, order].to_numpy(dtype=float)
    B = dist_b.loc[order, order].to_numpy(dtype=float)
    iu = np.triu_indices_from(A, k=1)
    a, b = A[iu], B[iu]
    r_obs = _corr(a, b, method)
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    ge = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        Bp = B[np.ix_(perm, perm)]
        if _corr(a, Bp[iu], method) >= r_obs:
            ge += 1
    p = (ge + 1) / (permutations + 1)
    return r_obs, p

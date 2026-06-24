"""Reference rhythm typology for the seed languages + agreement metrics.

Class labels follow the classical literature; Greek and Finnish are marked
'intermediate' (their classification is debated) and should be revisited in
exploration. Vocalic nPVI reference values are from Grabe & Low (2002) where
available. These are a validation *reference*, not ground truth.
"""

from __future__ import annotations

import math

from scipy.stats import spearmanr

RHYTHM_CLASS: dict[str, str] = {
    "english": "stress",
    "german": "stress",
    "polish": "stress",        # often classed stress-timed / mixed
    "french": "syllable",
    "spanish": "syllable",
    "italian": "syllable",
    "greek": "intermediate",
    "finnish": "intermediate",
}

# Verified vocalic nPVI values (Grabe & Low 2002). Subset only.
REFERENCE_NPVI_V: dict[str, float] = {
    "english": 57.2,
    "german": 59.7,
    "polish": 46.6,
    "french": 43.5,
    "spanish": 29.7,
}


def class_separation(values: dict[str, float]) -> float:
    """mean(stress-timed) - mean(syllable-timed) over labelled languages.

    Positive => stress-timed languages score higher (expected for vocalic nPVI).
    """
    stress = [v for k, v in values.items()
              if RHYTHM_CLASS.get(k) == "stress" and not math.isnan(v)]
    syllable = [v for k, v in values.items()
                if RHYTHM_CLASS.get(k) == "syllable" and not math.isnan(v)]
    if not stress or not syllable:
        return math.nan
    return sum(stress) / len(stress) - sum(syllable) / len(syllable)


def spearman_against_reference(values: dict[str, float]) -> float:
    """Spearman correlation between computed values and REFERENCE_NPVI_V."""
    shared = [k for k in REFERENCE_NPVI_V if k in values and not math.isnan(values[k])]
    if len(shared) < 3:
        return math.nan
    computed = [values[k] for k in shared]
    reference = [REFERENCE_NPVI_V[k] for k in shared]
    r, _ = spearmanr(computed, reference)
    return float(r)

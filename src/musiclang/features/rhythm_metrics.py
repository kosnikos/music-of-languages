"""Duration-based rhythm metrics (pure functions).

Each function documents the primary source for its maths in its own docstring.
General references (maths & reasoning):
  - Ramus, Nespor & Mehler (1999), Cognition 73:265-292 — %V, ΔC, ΔV:
    https://doi.org/10.1016/S0010-0277(99)00058-X
  - Grabe & Low (2002), Papers in Lab Phonology 7 — nPVI, rPVI:
    https://www.phon.ox.ac.uk/files/people/grabe/Grabe_Low.doc
  - Dellwo (2006), "A Variation Coefficient for ΔC" — VarcoC/VarcoV:
    https://www.lfsag.unito.it/sito_old/ritmo/varco_en.html
All inputs are lists of interval durations in seconds. NaN is returned where a
metric is undefined (too few intervals / empty input).
"""

from __future__ import annotations

import math
import statistics


def percent_v(vocalic: list[float], consonantal: list[float]) -> float:
    """Percentage of total interval duration that is vocalic (%V).

    Reference (maths & reasoning): Ramus, Nespor & Mehler (1999), Cognition 73:265-292,
    https://doi.org/10.1016/S0010-0277(99)00058-X
    """
    sum_v = sum(vocalic)
    total = sum_v + sum(consonantal)
    if total <= 0:
        return math.nan
    return 100.0 * sum_v / total


def delta(intervals: list[float]) -> float:
    """Population standard deviation of interval durations (ΔC / ΔV).

    Reference: Ramus, Nespor & Mehler (1999), https://doi.org/10.1016/S0010-0277(99)00058-X
    """
    if len(intervals) < 2:
        return math.nan
    return statistics.pstdev(intervals)


def varco(intervals: list[float]) -> float:
    """Rate-normalized delta: 100 * SD / mean (coefficient of variation).

    Reference: Dellwo (2006), "Rhythm and Speech Rate: A Variation Coefficient for ΔC";
    maths explained at https://www.lfsag.unito.it/sito_old/ritmo/varco_en.html
    """
    if len(intervals) < 2:
        return math.nan
    mean = statistics.fmean(intervals)
    if mean <= 0:
        return math.nan
    return 100.0 * statistics.pstdev(intervals) / mean


def npvi(intervals: list[float]) -> float:
    """Normalized Pairwise Variability Index (rate-normalized).

    Reference (formula): Grabe & Low (2002), Papers in Lab Phonology 7,
    https://www.phon.ox.ac.uk/files/people/grabe/Grabe_Low.doc
    """
    if len(intervals) < 2:
        return math.nan
    pairs = zip(intervals[:-1], intervals[1:])
    terms = [abs(a - b) / ((a + b) / 2.0) for a, b in pairs if (a + b) > 0]
    if not terms:
        return math.nan
    return 100.0 * statistics.fmean(terms)


def rpvi(intervals: list[float]) -> float:
    """Raw Pairwise Variability Index (not rate-normalized).

    Reference: Grabe & Low (2002), https://www.phon.ox.ac.uk/files/people/grabe/Grabe_Low.doc
    """
    if len(intervals) < 2:
        return math.nan
    diffs = [abs(a - b) for a, b in zip(intervals[:-1], intervals[1:])]
    return statistics.fmean(diffs)

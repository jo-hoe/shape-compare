"""Set-relative distance normalization and combination.

EFD and Hu distances have very different natural magnitudes, so a weighted sum of
the raw values is meaningless. We normalize each distance array to ``[0, 1]``
across the candidate set using a robust min-max, then combine with the
configured weights. Pure functions, trivially unit-tested.
"""

from __future__ import annotations

import numpy as np

# Percentiles for the robust min-max: clipping the extremes keeps a single
# outlier candidate from compressing every other distance toward zero.
_LOW_PCT = 5.0
_HIGH_PCT = 95.0


def normalize_distances(distances: np.ndarray) -> np.ndarray:
    """Scale a distance array to ``[0, 1]`` via a robust (percentile) min-max.

    A degenerate array (all equal, or a single element) maps to all zeros, i.e.
    "no information to discriminate on", which is the neutral choice.
    """
    values = np.asarray(distances, dtype=np.float64)
    if values.size < 2:
        return np.zeros_like(values)
    lo = float(np.percentile(values, _LOW_PCT))
    hi = float(np.percentile(values, _HIGH_PCT))
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def combine_distances(
    efd_norm: np.ndarray, hu_norm: np.ndarray, w_efd: float, w_hu: float
) -> np.ndarray:
    """Weighted mean of two normalized distance arrays (weights renormalized)."""
    total_weight = w_efd + w_hu
    if total_weight == 0.0:
        raise ValueError("at least one of w_efd, w_hu must be non-zero")
    combined: np.ndarray = (w_efd * efd_norm + w_hu * hu_norm) / total_weight
    return combined

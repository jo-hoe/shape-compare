"""Elliptic Fourier Descriptors (closed-form, scale/rotation invariant).

``pyefd`` with ``normalize=True`` removes translation, scale, rotation, and
start-point analytically, so there is no alignment search and therefore no local
optimum to fall into.
"""

from __future__ import annotations

import numpy as np
from pyefd import elliptic_fourier_descriptors


def compute_efd(contour: np.ndarray, harmonics: int) -> np.ndarray:
    """Return normalized EFD coefficients of shape ``(harmonics, 4)``."""
    coeffs = elliptic_fourier_descriptors(
        np.asarray(contour, dtype=np.float64),
        order=harmonics,
        normalize=True,
    )
    return np.asarray(coeffs, dtype=np.float64)


def efd_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two EFD coefficient sets, sign-ambiguity invariant.

    The first harmonic's ``a1``/``c1`` terms are sign-locked to ``1``/``0`` by
    ``pyefd``'s normalization, so they carry no information; dropping the whole
    first row keeps the metric meaningful and avoids a constant offset.

    ``pyefd``'s rotation normalization leaves one residual ambiguity: the
    canonical frame of a shape rotated near 180 degrees can settle into the
    antipodal orientation, which negates every coefficient. Comparing against
    both ``b`` and ``-b`` and taking the minimum resolves it in closed form --
    no search, so no local optimum.
    """
    tail_a = a[1:]
    tail_b = b[1:]
    return min(_l2(tail_a - tail_b), _l2(tail_a + tail_b))


def _l2(diff: np.ndarray) -> float:
    """Euclidean norm of a coefficient difference."""
    return float(np.sqrt(np.sum(diff * diff)))

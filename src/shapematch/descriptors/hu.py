"""Hu moment descriptor: a light, stable tie-breaker alongside EFD.

Hu's seven moment invariants are translation, scale, and rotation invariant by
construction. They are computed from the **filled silhouette mask** (not the
sparse contour), which makes the higher-order moments numerically stable. We
store them log-transformed so their wildly different magnitudes become
comparable. Hu is a coarse, noisy signal, so the engine weights it lightly and
uses it only to break ties between EFD-similar candidates.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_hu(mask: np.ndarray) -> np.ndarray:
    """Return the ``(7,)`` log-transformed Hu moments of a filled mask."""
    moments = cv2.moments(np.asarray(mask, dtype=np.uint8), binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    # -sign * log10(|hu|): compresses dynamic range, preserves sign information.
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = -np.sign(hu) * np.log10(np.abs(hu))
    return np.where(np.isfinite(logged), logged, 0.0).astype(np.float64)


def hu_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between two log-Hu vectors (cv2.matchShapes I1 semantics)."""
    return float(np.sum(np.abs(a - b)))

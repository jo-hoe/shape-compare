"""Optional Shape-Context re-rank with a global rotation sweep.

Shape context is more discriminative but costlier, so the engine only runs it on
the Stage-1 survivors. Rotation alignment is resolved by an **exhaustive sweep**
over a discrete set of angles, taking the global minimum cost -- never greedy
descent -- which is why this stage cannot be trapped in a local optimum.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def shape_context_histogram(
    points: np.ndarray, n_bins_r: int = 5, n_bins_theta: int = 12
) -> np.ndarray:
    """Compute per-point log-polar shape-context histograms.

    Returns an ``(M, n_bins_r * n_bins_theta)`` array: one normalized histogram
    of the relative positions of all other points, for each point.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    deltas = pts[:, None, :] - pts[None, :, :]
    r = np.sqrt(np.sum(deltas**2, axis=2))
    theta = (np.arctan2(deltas[:, :, 1], deltas[:, :, 0]) + 2 * np.pi) % (2 * np.pi)

    mean_r = np.mean(r[r > 0]) if np.any(r > 0) else 1.0
    r_norm = r / mean_r
    # Log-spaced radial bins; angular bins are uniform.
    r_edges = np.logspace(-1.0, 1.0, n_bins_r + 1)
    r_bin = np.clip(np.digitize(r_norm, r_edges), 0, n_bins_r - 1)
    theta_bin = np.clip((theta / (2 * np.pi) * n_bins_theta).astype(int), 0, n_bins_theta - 1)

    histograms = np.zeros((n, n_bins_r * n_bins_theta), dtype=np.float64)
    for i in range(n):
        flat = r_bin[i] * n_bins_theta + theta_bin[i]
        counts = np.bincount(np.delete(flat, i), minlength=n_bins_r * n_bins_theta)
        total = counts.sum()
        if total > 0:
            histograms[i] = counts / total
    return histograms


def sc_distance(points_a: np.ndarray, points_b: np.ndarray, rotation_steps: int) -> float:
    """Minimal shape-context cost over an exhaustive rotation sweep.

    ``points_a``/``points_b`` are PCA-normalized, arc-length-resampled contours.
    We rotate ``points_a`` through ``rotation_steps`` evenly spaced angles and
    return the smallest matching cost, i.e. the global optimum of the 1-D sweep.
    """
    hist_b = shape_context_histogram(points_b)
    best = float("inf")
    for angle in np.linspace(0.0, 2 * np.pi, rotation_steps, endpoint=False):
        rotated = _rotate(points_a, angle)
        cost = _matching_cost(shape_context_histogram(rotated), hist_b)
        best = min(best, cost)
    return best


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points about the origin by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ rot.T


def _matching_cost(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Greedy chi-square assignment cost between two histogram sets.

    A full Hungarian assignment is more accurate but unnecessary for a re-rank of
    a handful of survivors; the symmetric nearest-neighbour mean is stable and
    cheap.
    """
    cost = _chi_square(hist_a, hist_b)
    forward = cost.min(axis=1).mean()
    backward = cost.min(axis=0).mean()
    return float(0.5 * (forward + backward))


def _chi_square(hist_a: np.ndarray, hist_b: np.ndarray) -> np.ndarray:
    """Pairwise chi-square distance matrix between two sets of histograms."""
    matrix: np.ndarray = cdist(hist_a, hist_b, metric="sqeuclidean") * 0.5
    return matrix

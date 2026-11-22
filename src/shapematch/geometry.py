"""Pure geometry helpers: polygon/mask/contour conversion and normalization.

Every function here is pure (no I/O, no global state) and operates on NumPy
arrays, which keeps them trivial to unit-test and cheap to call from worker
processes. Rotation and scale are removed analytically via PCA; there is no
iterative alignment, so these helpers cannot land in a local optimum.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import CANVAS_PAD_FRAC, CANVAS_SIZE, MIN_CONTOUR_POINTS


def polygon_to_mask(polygon: np.ndarray, canvas_size: int = CANVAS_SIZE) -> np.ndarray:
    """Rasterize a polygon ring into a filled ``uint8`` silhouette mask.

    The polygon is translated and uniformly scaled (aspect ratio preserved) to
    fit a padded square canvas, then filled. Uniform scaling here keeps the
    rasterization faithful; scale normalization for matching happens later.
    """
    pts = _as_xy(polygon)
    fitted = _fit_to_canvas(pts, canvas_size)
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(fitted).astype(np.int32)], color=255)
    return mask


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Extract the single largest external contour from a mask as ``(N, 2)``.

    Raises:
        ValueError: If the mask contains no foreground contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("mask contains no contour")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float32)


def resample_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a closed contour to exactly ``n_points`` at uniform arc length."""
    pts = _as_xy(contour)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arclen[-1])
    if total == 0.0:
        raise ValueError("degenerate contour: zero perimeter")
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    x = np.interp(targets, arclen, closed[:, 0])
    y = np.interp(targets, arclen, closed[:, 1])
    return np.column_stack([x, y]).astype(np.float32)


def pca_normalize(contour: np.ndarray) -> np.ndarray:
    """Center, axis-align (PCA), and unit-scale a contour. Closed-form.

    Translation is removed by centering, rotation by rotating onto the principal
    axes, and scale by dividing by the RMS radius. The residual 180-degree / sign
    ambiguity of PCA is intentionally *not* guessed here; downstream EFD
    normalization and the mirror enumeration resolve it.
    """
    pts = _as_xy(contour).astype(np.float64)
    centered = pts - pts.mean(axis=0)
    # Principal axes are the eigenvectors of the covariance matrix.
    cov = np.cov(centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; put the major axis first.
    rotation = eigvecs[:, ::-1]
    aligned = centered @ rotation
    rms = float(np.sqrt(np.mean(np.sum(aligned**2, axis=1))))
    if rms == 0.0:
        raise ValueError("degenerate contour: zero extent")
    return (aligned / rms).astype(np.float32)


def aspect_rescale(contour: np.ndarray, ratios: tuple[float, ...]) -> np.ndarray:
    """Return ``(K, N, 2)`` copies of the contour scaled along the minor axis.

    Stretch tolerance is achieved by enumerating these discrete anisotropic
    scalings and letting the matcher take the best, rather than searching a
    continuous (and bumpy) stretch parameter.
    """
    pts = _as_xy(contour)
    variants = np.empty((len(ratios), pts.shape[0], 2), dtype=np.float32)
    for i, ratio in enumerate(ratios):
        scaled = pts.copy()
        scaled[:, 1] = scaled[:, 1] * ratio
        variants[i] = scaled
    return variants


def mirror_contour(contour: np.ndarray) -> np.ndarray:
    """Mirror a contour across the y-axis (flip x). Involution."""
    pts = _as_xy(contour).copy()
    pts[:, 0] = -pts[:, 0]
    return pts


def mirror_mask(mask: np.ndarray) -> np.ndarray:
    """Mirror a mask left-to-right (flip columns)."""
    return np.ascontiguousarray(np.fliplr(mask))


def contour_area(contour: np.ndarray) -> float:
    """Absolute polygon area via the shoelace formula."""
    pts = _as_xy(contour)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _as_xy(points: np.ndarray) -> np.ndarray:
    """Validate and coerce input into a contiguous ``(N, 2)`` float array."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {arr.shape}")
    if arr.shape[0] < MIN_CONTOUR_POINTS:
        raise ValueError(f"need at least {MIN_CONTOUR_POINTS} points, got {arr.shape[0]}")
    return arr


def _fit_to_canvas(points: np.ndarray, canvas_size: int) -> np.ndarray:
    """Translate+uniformly scale points into a padded square canvas."""
    lo = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lo, 1e-9)
    usable = canvas_size * (1.0 - 2.0 * CANVAS_PAD_FRAC)
    scale = usable / float(span.max())
    offset = canvas_size * CANVAS_PAD_FRAC
    fitted: np.ndarray = (points - lo) * scale + offset
    return fitted

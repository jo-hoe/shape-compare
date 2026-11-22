"""Core value objects shared across the shapematch pipeline.

All public data structures are immutable (`frozen=True`) dataclasses so they are
safe to pass between processes and cache without aliasing surprises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


@dataclass(frozen=True)
class Shape:
    """A normalized, filled silhouette ready for descriptor extraction.

    Attributes:
        mask: ``(H, W)`` ``uint8`` array, values in ``{0, 255}``; the filled
            silhouette rasterized once from the source polygon.
        contour: ``(N, 2)`` ``float32`` array; the single closed outer contour,
            counter-clockwise, re-extracted from ``mask`` for consistency.
        source_path: Origin file, or ``None`` for shapes built in-memory.
    """

    mask: np.ndarray
    contour: np.ndarray
    source_path: Path | None


@dataclass(frozen=True)
class Descriptor:
    """Precomputable, scale/rotation-invariant description of a single shape.

    Everything needed by both ranking stages is computed in one pass so that a
    candidate's descriptor can be cached and reused across queries. The aspect
    enumeration is baked into the EFD arrays; Hu is scale/rotation invariant and
    computed once from the filled mask (plus once from the mirrored mask).

    Attributes:
        efd: ``(K, harmonics, 4)`` normalized EFD per aspect variant.
        efd_mirror: ``(K, harmonics, 4)`` normalized EFD of each mirrored variant.
        hu: ``(7,)`` log-transformed Hu moments of the filled mask.
        hu_mirror: ``(7,)`` log-transformed Hu moments of the mirrored mask.
        aspect_ratios: ``(K,)`` the aspect ratio behind each variant index.
        contour_norm: ``(M, 2)`` PCA-normalized, arc-length-resampled points
            (the identity variant) used by the shape-context re-rank.
        meta: Small scalar metadata (harmonics, n_points).
    """

    efd: np.ndarray
    efd_mirror: np.ndarray
    hu: np.ndarray
    hu_mirror: np.ndarray
    aspect_ratios: np.ndarray
    contour_norm: np.ndarray
    meta: Mapping[str, float]


@dataclass(frozen=True)
class MatchResult:
    """The outcome of scoring one candidate against the target.

    Attributes:
        candidate: Path of the candidate shape file.
        score: Similarity in ``[0, 1]``; higher is more similar.
        efd_dist: Raw elliptic-Fourier distance at the winning orientation.
        hu_dist: Raw Hu-moment distance at the winning orientation.
        sc_dist: Shape-context distance if the re-rank stage ran, else ``None``.
        mirrored: ``True`` if the best match used the mirrored orientation.
        best_aspect: The aspect ratio that produced the best match.
    """

    candidate: Path
    score: float
    efd_dist: float
    hu_dist: float
    sc_dist: float | None
    mirrored: bool
    best_aspect: float

"""Assemble a full :class:`Descriptor` for a shape in a single pass.

Keeping descriptor assembly here (rather than in the engine) means a candidate's
entire descriptor -- per-aspect EFD/Hu (plus mirrors) and the normalized contour
-- is computed once and can be cached and reused across queries. The aspect
enumeration and mirror enumeration are both baked in so that matching never has
to recompute a descriptor.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import MatchParams
from ..geometry import (
    aspect_rescale,
    mirror_contour,
    mirror_mask,
    pca_normalize,
    resample_contour,
)
from ..types import Descriptor, Shape
from .efd import compute_efd, efd_distance
from .hu import compute_hu, hu_distance

__all__ = [
    "DescriptorComputer",
    "compute_descriptor",
    "compute_efd",
    "compute_hu",
    "efd_distance",
    "hu_distance",
]


class DescriptorComputer(Protocol):
    """Extension point: anything that turns a :class:`Shape` into a descriptor.

    Typing the engine against this Protocol lets alternative descriptor families
    (or non-geographic candidate domains) be swapped in without touching the
    ranking engine.
    """

    def __call__(self, shape: Shape, params: MatchParams) -> Descriptor: ...


def compute_descriptor(shape: Shape, params: MatchParams) -> Descriptor:
    """Build the full descriptor used by both ranking stages."""
    resampled = resample_contour(shape.contour, params.n_points)
    normalized = pca_normalize(resampled)
    variants = aspect_rescale(normalized, params.aspect_ratios)

    efd = np.stack([compute_efd(v, params.harmonics) for v in variants])
    efd_mirror = np.stack([compute_efd(mirror_contour(v), params.harmonics) for v in variants])

    return Descriptor(
        efd=efd,
        efd_mirror=efd_mirror,
        hu=compute_hu(shape.mask),
        hu_mirror=compute_hu(mirror_mask(shape.mask)),
        aspect_ratios=np.asarray(params.aspect_ratios, dtype=np.float64),
        contour_norm=normalized,
        meta={
            "harmonics": float(params.harmonics),
            "n_points": float(params.n_points),
        },
    )

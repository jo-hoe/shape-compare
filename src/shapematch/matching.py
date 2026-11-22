"""Pairwise scoring of a target against one candidate descriptor.

Stretch and mirror tolerance are handled by **enumeration**: we compare the
target (at its identity aspect) against every aspect variant of the candidate,
optionally including the mirrored variants, and take the orientation with the
smallest elliptic-Fourier distance. There is no optimizer and therefore no
local-minimum trap.

This function returns *raw* component distances only. Combining EFD and Hu into
a single score requires normalizing each across the whole candidate set (their
natural scales differ by orders of magnitude), which is the engine's job.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import MatchParams
from .descriptors import efd_distance, hu_distance
from .types import Descriptor, MatchResult

# The target is always compared at its natural aspect; only the candidate is
# stretched, so stretch tolerance is applied once (not squared).
_IDENTITY_ASPECT_INDEX = 0


def pairwise_score(
    candidate_path: Path,
    target: Descriptor,
    candidate: Descriptor,
    params: MatchParams,
) -> MatchResult:
    """Score ``candidate`` against ``target``, returning raw component distances.

    The winning orientation is the aspect/mirror combination with the smallest
    EFD distance (EFD is the primary, well-scaled invariant). ``score`` is left
    at ``0.0`` here and filled in by the engine once distances are normalized
    across all candidates.
    """
    best = _best_orientation(target, candidate, params)
    return MatchResult(
        candidate=candidate_path,
        score=0.0,
        efd_dist=best.efd,
        hu_dist=best.hu,
        sc_dist=None,
        mirrored=best.mirrored,
        best_aspect=best.aspect,
    )


class _Orientation:
    """The winning (aspect, mirror) combination and its raw distances."""

    __slots__ = ("aspect", "efd", "hu", "mirrored")

    def __init__(self, efd: float, hu: float, aspect: float, mirrored: bool) -> None:
        self.efd = efd
        self.hu = hu
        self.aspect = aspect
        self.mirrored = mirrored


def _best_orientation(
    target: Descriptor, candidate: Descriptor, params: MatchParams
) -> _Orientation:
    """Return the minimum-EFD-distance orientation over the discrete grid."""
    t_efd = target.efd[_IDENTITY_ASPECT_INDEX]
    t_hu = target.hu
    ratios = candidate.aspect_ratios

    best = _score_variants(t_efd, t_hu, candidate.efd, candidate.hu, ratios, mirrored=False)
    if params.allow_mirror:
        mirror = _score_variants(
            t_efd, t_hu, candidate.efd_mirror, candidate.hu_mirror, ratios, mirrored=True
        )
        if mirror.efd < best.efd:
            best = mirror
    return best


def _score_variants(
    t_efd: np.ndarray,
    t_hu: np.ndarray,
    cand_efd: np.ndarray,
    cand_hu: np.ndarray,
    aspect_ratios: np.ndarray,
    *,
    mirrored: bool,
) -> _Orientation:
    """Find the minimum-EFD aspect variant for one orientation.

    Hu is scale/rotation invariant and not aspect-varied, so a single Hu distance
    applies across all aspect variants of this orientation.
    """
    hu = hu_distance(t_hu, cand_hu)
    best: _Orientation | None = None
    for i in range(cand_efd.shape[0]):
        orient = _Orientation(
            efd=efd_distance(t_efd, cand_efd[i]),
            hu=hu,
            aspect=float(aspect_ratios[i]),
            mirrored=mirrored,
        )
        if best is None or orient.efd < best.efd:
            best = orient
    assert best is not None  # aspect_ratios is always non-empty
    return best

"""Matching configuration and the canvas/geometry constants.

``MatchParams`` is frozen and fully typed; every field that affects a computed
descriptor is folded into :meth:`MatchParams.cache_key` so cached descriptors
invalidate automatically when the parameters change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Fixed raster canvas used when rasterizing a polygon to a silhouette mask.
# Large enough to preserve fine coastline detail, small enough to stay cheap.
CANVAS_SIZE: int = 512

# Fraction of the canvas left as a margin around the shape when rasterizing.
CANVAS_PAD_FRAC: float = 0.05

# Guard against degenerate input: a contour needs at least a triangle.
MIN_CONTOUR_POINTS: int = 3


@dataclass(frozen=True)
class MatchParams:
    """Tunable parameters for descriptor computation and ranking.

    Defaults match the approved plan. ``allow_mirror`` defaults to ``False`` so a
    mirrored shape counts as a *different* shape unless explicitly enabled.
    """

    harmonics: int = 20
    n_points: int = 128
    aspect_ratios: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25)
    allow_mirror: bool = False
    use_shape_context: bool = False
    sc_rotation_steps: int = 36
    w_efd: float = 0.9
    w_hu: float = 0.1
    w_sc: float = 1.0
    top_k: int = 20
    stage1_keep: int = 50

    # Derived once; excluded from equality/replace semantics callers care about.
    _key: str = field(default="", compare=False, repr=False)

    def cache_key(self) -> str:
        """Return a stable hash of every descriptor-affecting field.

        Ranking-only fields (``top_k``, ``stage1_keep``, weights) are excluded:
        they do not change a stored :class:`~shapematch.types.Descriptor`, so
        tweaking them must not invalidate the cache.
        """
        payload = "|".join(
            str(value)
            for value in (
                self.harmonics,
                self.n_points,
                self.aspect_ratios,
                self.sc_rotation_steps,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

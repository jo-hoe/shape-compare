"""The ranking engine: a two-stage pipeline over all candidates.

Stage 1 is a fast, parallel, closed-form EFD+Hu filter across every candidate.
Stage 2 (optional) re-ranks only the Stage-1 survivors with the costlier
shape-context distance. Dependencies (cache, executor) are injected so the engine
stays easy to test and reconfigure.

EFD and Hu distances live on very different natural scales, so they are
normalized across the candidate set (robust min-max) before being combined into
a single similarity. This set-relative normalization is why scoring happens here,
after all raw distances are known, rather than per-pair.
"""

from __future__ import annotations

import operator
from pathlib import Path

import numpy as np

from .cache import DescriptorCache
from .config import MatchParams
from .descriptors import compute_descriptor
from .descriptors.shape_context import sc_distance
from .loader import load_polygon
from .parallel import ParallelExecutor
from .scoring import combine_distances, normalize_distances
from .types import Descriptor, MatchResult, Shape


class RankingEngine:
    """Ranks candidate shape files against a target shape."""

    def __init__(
        self,
        params: MatchParams,
        executor: ParallelExecutor,
        cache: DescriptorCache | None,
    ) -> None:
        self._params = params
        self._executor = executor
        self._cache = cache

    def rank(self, target: Shape, candidate_paths: list[Path]) -> list[MatchResult]:
        """Return the top-``k`` matches, best first."""
        if not candidate_paths:
            return []
        target_desc = compute_descriptor(target, self._params)
        cache_dir = self._cache.directory if self._cache is not None else None

        raw = self._executor.map_scores(target_desc, candidate_paths, self._params, cache_dir)
        stage1 = self._score_stage1(raw)
        stage1.sort(key=operator.attrgetter("score"), reverse=True)

        if self._params.use_shape_context:
            survivors = stage1[: self._params.stage1_keep]
            reranked = self._shape_context_rerank(target_desc, survivors)
            return reranked[: self._params.top_k]
        return stage1[: self._params.top_k]

    def _score_stage1(self, raw: list[MatchResult]) -> list[MatchResult]:
        """Normalize EFD/Hu across the set and fold them into a similarity."""
        efd_norm = normalize_distances(np.array([r.efd_dist for r in raw]))
        hu_norm = normalize_distances(np.array([r.hu_dist for r in raw]))
        combined = combine_distances(efd_norm, hu_norm, self._params.w_efd, self._params.w_hu)
        return [
            _with_score(result, score=1.0 - float(dist))
            for result, dist in zip(raw, combined, strict=True)
        ]

    def _shape_context_rerank(
        self, target: Descriptor, survivors: list[MatchResult]
    ) -> list[MatchResult]:
        """Blend a normalized shape-context distance into survivors and re-sort."""
        sc_raw = np.array([self._sc_for(target, r.candidate) for r in survivors])
        sc_norm = normalize_distances(sc_raw)
        base = np.array([1.0 - r.score for r in survivors])  # recover combined distance
        total = self._params.w_sc * sc_norm + base
        reranked = [
            _with_sc(result, sc=float(raw_sc), score=1.0 - float(dist))
            for result, raw_sc, dist in zip(survivors, sc_raw, total, strict=True)
        ]
        reranked.sort(key=operator.attrgetter("score"), reverse=True)
        return reranked

    def _sc_for(self, target: Descriptor, path: Path) -> float:
        """Shape-context distance between the target and one candidate."""
        candidate = self._load_descriptor(path)
        return sc_distance(
            target.contour_norm, candidate.contour_norm, self._params.sc_rotation_steps
        )

    def _load_descriptor(self, path: Path) -> Descriptor:
        """Reload a candidate descriptor (cache-backed when enabled)."""
        if self._cache is None:
            return compute_descriptor(load_polygon(path), self._params)
        return self._cache.get_or_compute(path, self._params, load_polygon, compute_descriptor)


def _with_score(result: MatchResult, *, score: float) -> MatchResult:
    """Return a copy of ``result`` with the final similarity score set."""
    return MatchResult(
        candidate=result.candidate,
        score=score,
        efd_dist=result.efd_dist,
        hu_dist=result.hu_dist,
        sc_dist=result.sc_dist,
        mirrored=result.mirrored,
        best_aspect=result.best_aspect,
    )


def _with_sc(result: MatchResult, *, sc: float, score: float) -> MatchResult:
    """Return a copy of ``result`` with the shape-context distance and score set."""
    return MatchResult(
        candidate=result.candidate,
        score=score,
        efd_dist=result.efd_dist,
        hu_dist=result.hu_dist,
        sc_dist=sc,
        mirrored=result.mirrored,
        best_aspect=result.best_aspect,
    )

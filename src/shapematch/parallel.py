"""Parallel candidate scoring over a process pool.

Descriptor computation for independent candidates is embarrassingly parallel.
Workers each load/compute (via the cache) a candidate descriptor and score it
against the shared target descriptor. ``workers == 1`` runs serially in-process,
which keeps debugging and tests deterministic; serial and parallel paths produce
identical rankings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from .cache import DescriptorCache
from .config import MatchParams
from .descriptors import compute_descriptor
from .loader import load_polygon
from .matching import pairwise_score
from .types import Descriptor, MatchResult


@dataclass(frozen=True)
class _Job:
    """Everything a worker needs to score one candidate. Must be picklable."""

    candidate_path: Path
    target: Descriptor
    params: MatchParams
    cache_dir: Path | None


class ParallelExecutor:
    """Runs :func:`pairwise_score` across candidates, serially or pooled."""

    def __init__(self, workers: int | None = None) -> None:
        self._workers = workers if workers is not None else (os.cpu_count() or 1)

    def map_scores(
        self,
        target: Descriptor,
        candidate_paths: list[Path],
        params: MatchParams,
        cache_dir: Path | None,
    ) -> list[MatchResult]:
        """Score every candidate and return results (unordered by caller)."""
        jobs = [_Job(path, target, params, cache_dir) for path in candidate_paths]
        if self._workers == 1:
            return [_run_job(job) for job in jobs]
        chunksize = max(1, len(jobs) // (self._workers * 4))
        with Pool(processes=self._workers) as pool:
            return list(pool.imap_unordered(_run_job, jobs, chunksize=chunksize))


def _run_job(job: _Job) -> MatchResult:
    """Top-level worker function (picklable): compute + score one candidate."""
    candidate = _candidate_descriptor(job)
    return pairwise_score(job.candidate_path, job.target, candidate, job.params)


def _candidate_descriptor(job: _Job) -> Descriptor:
    """Resolve a candidate's descriptor, using the cache when enabled."""
    if job.cache_dir is None:
        return compute_descriptor(load_polygon(job.candidate_path), job.params)
    cache = DescriptorCache(job.cache_dir)
    return cache.get_or_compute(job.candidate_path, job.params, load_polygon, compute_descriptor)

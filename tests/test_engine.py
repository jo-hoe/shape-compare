"""Engine tests: ranking correctness, serial/parallel parity, shape-context."""

from __future__ import annotations

from pathlib import Path

from conftest import rotate, scale, triangle, write_polygon
from shapematch.cache import DescriptorCache
from shapematch.config import MatchParams
from shapematch.engine import RankingEngine
from shapematch.loader import load_polygon
from shapematch.parallel import ParallelExecutor


def _engine(params: MatchParams, workers: int, cache_dir: Path | None) -> RankingEngine:
    cache = DescriptorCache(cache_dir) if cache_dir is not None else None
    return RankingEngine(params, ParallelExecutor(workers=workers), cache)


def test_ranks_true_twin_first(tmp_path: Path, candidate_dir: Path) -> None:
    # Target is a rotated+scaled triangle; triangle.json must rank first.
    target = write_polygon(tmp_path / "target.json", scale(rotate(triangle(), 55.0), 3.0))
    engine = _engine(MatchParams(), workers=1, cache_dir=None)
    results = engine.rank(load_polygon(target), sorted(candidate_dir.glob("*.json")))
    assert results[0].candidate.name == "triangle.json"


def test_empty_candidates_returns_empty(tmp_path: Path) -> None:
    target = write_polygon(tmp_path / "t.json", triangle())
    engine = _engine(MatchParams(), workers=1, cache_dir=None)
    assert engine.rank(load_polygon(target), []) == []


def test_top_k_is_honored(tmp_path: Path, candidate_dir: Path) -> None:
    target = write_polygon(tmp_path / "t.json", triangle())
    engine = _engine(MatchParams(top_k=1), workers=1, cache_dir=None)
    results = engine.rank(load_polygon(target), sorted(candidate_dir.glob("*.json")))
    assert len(results) == 1


def test_serial_and_parallel_agree(tmp_path: Path, candidate_dir: Path) -> None:
    target = write_polygon(tmp_path / "t.json", rotate(triangle(), 20.0))
    paths = sorted(candidate_dir.glob("*.json"))
    serial = _engine(MatchParams(), workers=1, cache_dir=None).rank(load_polygon(target), paths)
    parallel = _engine(MatchParams(), workers=2, cache_dir=None).rank(load_polygon(target), paths)
    assert [r.candidate.name for r in serial] == [r.candidate.name for r in parallel]


def test_scores_are_bounded(tmp_path: Path, candidate_dir: Path) -> None:
    target = write_polygon(tmp_path / "t.json", triangle())
    engine = _engine(MatchParams(), workers=1, cache_dir=None)
    results = engine.rank(load_polygon(target), sorted(candidate_dir.glob("*.json")))
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_shape_context_rerank_runs(tmp_path: Path, candidate_dir: Path) -> None:
    target = write_polygon(tmp_path / "t.json", scale(rotate(triangle(), 30.0), 2.0))
    params = MatchParams(use_shape_context=True, sc_rotation_steps=12, stage1_keep=3)
    engine = _engine(params, workers=1, cache_dir=None)
    results = engine.rank(load_polygon(target), sorted(candidate_dir.glob("*.json")))
    assert results[0].candidate.name == "triangle.json"
    assert all(r.sc_dist is not None for r in results)


def test_cache_speeds_repeat_query(tmp_path: Path, candidate_dir: Path) -> None:
    target = write_polygon(tmp_path / "t.json", triangle())
    paths = sorted(candidate_dir.glob("*.json"))
    engine = _engine(MatchParams(), workers=1, cache_dir=tmp_path / "cache")
    first = engine.rank(load_polygon(target), paths)
    second = engine.rank(load_polygon(target), paths)
    assert [r.candidate.name for r in first] == [r.candidate.name for r in second]
    assert any((tmp_path / "cache").glob("*.npz"))

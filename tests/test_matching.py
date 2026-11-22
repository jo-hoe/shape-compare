"""Matching tests: orientation selection, mirror flag behavior."""

from __future__ import annotations

from pathlib import Path

from conftest import mirror, rotate, triangle, write_polygon
from shapematch.config import MatchParams
from shapematch.descriptors import compute_descriptor
from shapematch.loader import load_polygon
from shapematch.matching import pairwise_score


def _descriptor(tmp_path: Path, points, name: str, params: MatchParams):  # type: ignore[no-untyped-def]
    path = write_polygon(tmp_path / f"{name}.json", points)
    return compute_descriptor(load_polygon(path), params)


def test_identical_shape_has_near_zero_efd(tmp_path: Path) -> None:
    params = MatchParams()
    target = _descriptor(tmp_path, triangle(), "t", params)
    candidate = _descriptor(tmp_path, rotate(triangle(), 40.0), "c", params)
    result = pairwise_score(Path("c.json"), target, candidate, params)
    assert result.efd_dist < 0.05
    assert not result.mirrored


def test_mirror_disabled_does_not_report_mirror(tmp_path: Path) -> None:
    params = MatchParams(allow_mirror=False)
    target = _descriptor(tmp_path, triangle(), "t", params)
    mirrored = _descriptor(tmp_path, mirror(triangle()), "m", params)
    result = pairwise_score(Path("m.json"), target, mirrored, params)
    assert result.mirrored is False


def test_mirror_enabled_matches_mirrored_shape(tmp_path: Path) -> None:
    params = MatchParams(allow_mirror=True)
    target = _descriptor(tmp_path, triangle(), "t", params)
    mirrored = _descriptor(tmp_path, mirror(triangle()), "m", params)

    disabled = pairwise_score(Path("m.json"), target, mirrored, MatchParams(allow_mirror=False))
    enabled = pairwise_score(Path("m.json"), target, mirrored, params)

    # Allowing mirror must not worsen the match of a mirrored asymmetric shape.
    assert enabled.efd_dist <= disabled.efd_dist
    assert enabled.mirrored is True
    assert enabled.efd_dist < 0.05


def test_best_aspect_is_reported(tmp_path: Path) -> None:
    params = MatchParams()
    target = _descriptor(tmp_path, triangle(), "t", params)
    candidate = _descriptor(tmp_path, triangle(), "c", params)
    result = pairwise_score(Path("c.json"), target, candidate, params)
    assert result.best_aspect in params.aspect_ratios

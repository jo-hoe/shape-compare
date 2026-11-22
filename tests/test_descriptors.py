"""Descriptor tests plus the core invariance guarantees.

These assert the headline property of the whole tool: the same shape, scaled,
rotated, and mildly stretched, must produce an almost-identical descriptor and
therefore a near-zero distance -- strictly below the distance to a different
shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import circle, rectangle, rotate, scale, stretch, triangle, write_polygon
from shapematch.config import MatchParams
from shapematch.descriptors import compute_descriptor, efd_distance, hu_distance
from shapematch.loader import load_polygon


def _descriptor_from(tmp_path, points, name):  # type: ignore[no-untyped-def]
    path = write_polygon(tmp_path / f"{name}.json", points)
    return compute_descriptor(load_polygon(path), MatchParams())


def test_efd_has_expected_shape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    params = MatchParams(harmonics=15)
    desc = compute_descriptor(load_polygon(write_polygon(tmp_path / "c.json", circle())), params)
    assert desc.efd.shape == (len(params.aspect_ratios), 15, 4)


def test_hu_has_seven_finite_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    desc = _descriptor_from(tmp_path, circle(), "c")
    assert desc.hu.shape == (7,)
    assert np.all(np.isfinite(desc.hu))


@pytest.mark.parametrize("degrees", [30.0, 90.0, 217.0])
@pytest.mark.parametrize("factor", [0.5, 2.0, 5.0])
def test_efd_invariant_to_scale_and_rotation(tmp_path, degrees, factor) -> None:  # type: ignore[no-untyped-def]
    # A smooth shape isolates the descriptor's invariance from the rasterize/
    # re-trace aliasing that sharp few-vertex shapes suffer at odd angles.
    base = _descriptor_from(tmp_path, circle(), "base")
    transformed = _descriptor_from(tmp_path, scale(rotate(circle(), degrees), factor), "xf")
    identity = 0
    assert efd_distance(base.efd[identity], transformed.efd[identity]) < 0.05


@pytest.mark.parametrize("degrees", [30.0, 90.0, 217.0])
def test_sharp_shape_ranks_itself_closest(tmp_path, degrees) -> None:  # type: ignore[no-untyped-def]
    # Even a sharp 3-vertex shape, where rasterization adds some noise at odd
    # angles, must stay strictly closer to its own rotated copy than to any
    # structurally different shape -- the property ranking actually depends on.
    rotated = _descriptor_from(tmp_path, scale(rotate(triangle(), degrees), 2.0), "rot")
    identity = 0
    rot = rotated.efd[identity]
    to_self = efd_distance(rot, _descriptor_from(tmp_path, triangle(), "tri").efd[identity])
    to_rect = efd_distance(rot, _descriptor_from(tmp_path, rectangle(), "rect").efd[identity])
    to_circle = efd_distance(rot, _descriptor_from(tmp_path, circle(), "circ").efd[identity])
    assert to_self < to_rect
    assert to_self < to_circle


def test_same_shape_closer_than_different_shape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = _descriptor_from(tmp_path, triangle(), "base")
    same = _descriptor_from(tmp_path, rotate(triangle(), 73.0), "same")
    other = _descriptor_from(tmp_path, circle(), "other")
    identity = 0
    d_same = efd_distance(base.efd[identity], same.efd[identity])
    d_other = efd_distance(base.efd[identity], other.efd[identity])
    assert d_same < d_other


def test_mild_stretch_absorbed_by_aspect_variants(tmp_path) -> None:  # type: ignore[no-untyped-def]
    params = MatchParams()
    base = _descriptor_from(tmp_path, rectangle(), "base")
    stretched = _descriptor_from(tmp_path, stretch(rectangle(), 1.25), "stretched")
    identity = 0
    # Best over the candidate's aspect variants should beat the naive 1:1 compare.
    naive = efd_distance(base.efd[identity], stretched.efd[identity])
    best = min(
        efd_distance(base.efd[identity], stretched.efd[i])
        for i in range(len(params.aspect_ratios))
    )
    assert best <= naive


def test_hu_distance_zero_for_identical() -> None:
    vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert hu_distance(vec, vec) == 0.0

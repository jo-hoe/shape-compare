"""Geometry tests: conversion, resampling, PCA normalization, mirror, area."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import circle, rectangle, rotate, scale
from shapematch.geometry import (
    aspect_rescale,
    contour_area,
    mask_to_contour,
    mirror_contour,
    mirror_mask,
    pca_normalize,
    polygon_to_mask,
    resample_contour,
)


def test_polygon_to_mask_is_binary() -> None:
    mask = polygon_to_mask(circle())
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask.max() == 255


def test_mask_to_contour_roundtrip() -> None:
    mask = polygon_to_mask(rectangle())
    contour = mask_to_contour(mask)
    assert contour.ndim == 2
    assert contour.shape[1] == 2
    assert len(contour) >= 3


def test_mask_to_contour_empty_raises() -> None:
    with pytest.raises(ValueError, match="no contour"):
        mask_to_contour(np.zeros((10, 10), dtype=np.uint8))


def test_resample_returns_exact_count() -> None:
    out = resample_contour(circle(n=37), 128)
    assert out.shape == (128, 2)


def test_resample_zero_perimeter_raises() -> None:
    degenerate = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="zero perimeter"):
        resample_contour(degenerate, 10)


def test_pca_normalize_is_centered_and_unit_scale() -> None:
    out = pca_normalize(rectangle())
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-5)
    rms = np.sqrt(np.mean(np.sum(out**2, axis=1)))
    assert rms == pytest.approx(1.0, abs=1e-5)


def test_pca_normalize_invariant_to_rotation_and_scale() -> None:
    base = pca_normalize(rectangle())
    transformed = pca_normalize(scale(rotate(rectangle(), 50.0), 4.0))
    # Allow axis sign flips: compare the set of absolute coordinates.
    a = np.sort(np.abs(base), axis=0)
    b = np.sort(np.abs(transformed), axis=0)
    assert np.allclose(a, b, atol=1e-2)


def test_pca_normalize_degenerate_raises() -> None:
    with pytest.raises(ValueError, match="zero extent"):
        pca_normalize(np.zeros((5, 2), dtype=np.float32))


def test_aspect_rescale_shape() -> None:
    normalized = pca_normalize(circle())
    variants = aspect_rescale(normalized, (0.8, 1.0, 1.25))
    assert variants.shape == (3, normalized.shape[0], 2)
    # Ratio 1.0 is the identity.
    identity_index = 1
    assert np.allclose(variants[identity_index], normalized)


def test_mirror_contour_is_involution() -> None:
    pts = circle()
    assert np.allclose(mirror_contour(mirror_contour(pts)), pts)


def test_mirror_mask_flips_columns() -> None:
    mask = polygon_to_mask(rectangle())
    assert np.array_equal(mirror_mask(mirror_mask(mask)), mask)


def test_contour_area_of_unit_square() -> None:
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    assert contour_area(square) == pytest.approx(1.0)


def test_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        resample_contour(np.zeros((5, 3), dtype=np.float32), 10)

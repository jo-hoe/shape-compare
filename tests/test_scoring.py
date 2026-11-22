"""Scoring tests: robust normalization and weighted combination."""

from __future__ import annotations

import numpy as np
import pytest

from shapematch.scoring import combine_distances, normalize_distances


def test_normalize_maps_to_unit_range() -> None:
    out = normalize_distances(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_normalize_single_element_is_zero() -> None:
    assert np.array_equal(normalize_distances(np.array([42.0])), np.array([0.0]))


def test_normalize_all_equal_is_zero() -> None:
    out = normalize_distances(np.full(5, 7.0))
    assert np.array_equal(out, np.zeros(5))


def test_normalize_clips_outliers() -> None:
    # An extreme outlier must not push the bulk toward zero.
    out = normalize_distances(np.array([0.0, 1.0, 1.0, 1.0, 1000.0]))
    assert out.max() == pytest.approx(1.0)


def test_combine_is_weighted_mean() -> None:
    efd = np.array([0.0, 1.0])
    hu = np.array([1.0, 0.0])
    out = combine_distances(efd, hu, w_efd=0.9, w_hu=0.1)
    assert out[0] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.9)


def test_combine_zero_weights_raises() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        combine_distances(np.array([1.0]), np.array([1.0]), w_efd=0.0, w_hu=0.0)

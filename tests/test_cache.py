"""Cache tests: round-trip, invalidation, and atomic writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import circle, write_polygon
from shapematch.cache import DescriptorCache
from shapematch.config import MatchParams
from shapematch.descriptors import compute_descriptor
from shapematch.loader import load_polygon


def test_put_get_roundtrip(tmp_path: Path) -> None:
    params = MatchParams()
    path = write_polygon(tmp_path / "c.json", circle())
    desc = compute_descriptor(load_polygon(path), params)
    cache = DescriptorCache(tmp_path / "cache")

    assert cache.get(path, params) is None
    cache.put(path, params, desc)
    loaded = cache.get(path, params)

    assert loaded is not None
    assert np.allclose(loaded.efd, desc.efd)
    assert np.allclose(loaded.hu, desc.hu)
    assert np.allclose(loaded.aspect_ratios, desc.aspect_ratios)


def test_param_change_invalidates(tmp_path: Path) -> None:
    path = write_polygon(tmp_path / "c.json", circle())
    cache = DescriptorCache(tmp_path / "cache")
    params_a = MatchParams(harmonics=10)
    params_b = MatchParams(harmonics=20)

    cache.put(path, params_a, compute_descriptor(load_polygon(path), params_a))
    assert cache.get(path, params_b) is None


def test_file_change_invalidates(tmp_path: Path) -> None:
    params = MatchParams()
    path = write_polygon(tmp_path / "c.json", circle(r=1.0))
    cache = DescriptorCache(tmp_path / "cache")
    cache.put(path, params, compute_descriptor(load_polygon(path), params))

    write_polygon(path, circle(r=2.0, n=150))  # different content -> different hash
    assert cache.get(path, params) is None


def test_get_or_compute_populates_cache(tmp_path: Path) -> None:
    params = MatchParams()
    path = write_polygon(tmp_path / "c.json", circle())
    cache = DescriptorCache(tmp_path / "cache")

    first = cache.get_or_compute(path, params, load_polygon, compute_descriptor)
    assert cache.get(path, params) is not None
    second = cache.get_or_compute(path, params, load_polygon, compute_descriptor)
    assert np.allclose(first.efd, second.efd)


def test_directory_property(tmp_path: Path) -> None:
    cache = DescriptorCache(tmp_path / "cache")
    assert cache.directory == tmp_path / "cache"
    assert cache.directory.is_dir()

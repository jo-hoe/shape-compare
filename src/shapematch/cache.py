"""On-disk descriptor cache keyed by file content hash + descriptor params.

Each candidate descriptor is stored as a compressed ``.npz``. The key combines
the file's content hash with the descriptor-affecting parameter hash, so editing
a candidate file or changing descriptor parameters transparently invalidates the
entry, while repeated queries with the same inputs are instant.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import MatchParams
from .types import Descriptor

if TYPE_CHECKING:
    from .descriptors import DescriptorComputer
    from .loader import PolygonLoader


class DescriptorCache:
    """Reads/writes :class:`Descriptor` objects to a cache directory."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        """The directory backing this cache."""
        return self._dir

    def key(self, path: Path, params: MatchParams) -> str:
        """Return ``<file-content-hash>_<params-hash>`` for ``path``."""
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
        return f"{digest}_{params.cache_key()}"

    def get(self, path: Path, params: MatchParams) -> Descriptor | None:
        """Load a cached descriptor, or ``None`` on miss."""
        cache_path = self._dir / f"{self.key(path, params)}.npz"
        if not cache_path.exists():
            return None
        return _descriptor_from_npz(cache_path)

    def put(self, path: Path, params: MatchParams, descriptor: Descriptor) -> None:
        """Atomically store a descriptor (temp file + rename, worker-safe)."""
        cache_path = self._dir / f"{self.key(path, params)}.npz"
        _atomic_savez(cache_path, descriptor)

    def get_or_compute(
        self,
        path: Path,
        params: MatchParams,
        loader: PolygonLoader,
        computer: DescriptorComputer,
    ) -> Descriptor:
        """Return a cached descriptor or compute, store, and return a fresh one."""
        cached = self.get(path, params)
        if cached is not None:
            return cached
        descriptor = computer(loader(path), params)
        self.put(path, params, descriptor)
        return descriptor


def _descriptor_from_npz(cache_path: Path) -> Descriptor:
    """Reconstruct a :class:`Descriptor` from an ``.npz`` archive."""
    with np.load(cache_path, allow_pickle=False) as data:
        return Descriptor(
            efd=data["efd"],
            efd_mirror=data["efd_mirror"],
            hu=data["hu"],
            hu_mirror=data["hu_mirror"],
            aspect_ratios=data["aspect_ratios"],
            contour_norm=data["contour_norm"],
            meta={"harmonics": float(data["harmonics"]), "n_points": float(data["n_points"])},
        )


def _atomic_savez(cache_path: Path, descriptor: Descriptor) -> None:
    """Write the ``.npz`` to a temp file in the same dir, then atomic-rename."""
    fd, tmp_name = tempfile.mkstemp(dir=cache_path.parent, suffix=".npz.tmp")
    tmp_path = Path(tmp_name)
    try:
        # Write through the open handle so numpy does not append its own
        # ``.npz`` suffix to the temp name (which would leave the mkstemp file
        # empty and corrupt the cache entry).
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                efd=descriptor.efd,
                efd_mirror=descriptor.efd_mirror,
                hu=descriptor.hu,
                hu_mirror=descriptor.hu_mirror,
                aspect_ratios=descriptor.aspect_ratios,
                contour_norm=descriptor.contour_norm,
                harmonics=descriptor.meta["harmonics"],
                n_points=descriptor.meta["n_points"],
            )
        os.replace(tmp_path, cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)

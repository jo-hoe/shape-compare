"""shape-compare: match a target silhouette against candidate shapes.

Public API re-exports the stable building blocks so library users can import
from the package root rather than reaching into submodules.
"""

from __future__ import annotations

from .cache import DescriptorCache
from .config import MatchParams
from .descriptors import compute_descriptor
from .engine import RankingEngine
from .loader import load_polygon
from .parallel import ParallelExecutor
from .types import Descriptor, MatchResult, Shape

__version__ = "0.1.0"

__all__ = [
    "DescriptorCache",
    "Descriptor",
    "MatchParams",
    "MatchResult",
    "ParallelExecutor",
    "RankingEngine",
    "Shape",
    "compute_descriptor",
    "load_polygon",
]

"""The single input path: a JSON polygon file to a :class:`Shape`.

The app accepts exactly one format and performs no normalization across formats.
Input is treated as untrusted: it is parsed with the stdlib ``json`` module (no
code execution), the schema is validated strictly, and the point count is capped
to avoid pathological rasterization cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import MIN_CONTOUR_POINTS
from .geometry import mask_to_contour, polygon_to_mask
from .types import Shape

# Upper bound on input polygon vertices (defensive against huge/hostile files).
MAX_CONTOUR_POINTS: int = 2_000_000


class PolygonLoader(Protocol):
    """Extension point: anything that turns a file path into a :class:`Shape`."""

    def __call__(self, path: Path) -> Shape: ...


def load_polygon(path: Path) -> Shape:
    """Load a one-polygon JSON file and return a normalized :class:`Shape`.

    The expected schema is ``{"contour": [[x, y], ...]}`` with at least three
    numeric coordinate pairs. The polygon is rasterized once to a filled mask and
    the canonical contour is re-extracted from that mask, guaranteeing the mask
    and contour are always consistent.

    Raises:
        ValueError: If the file is not valid JSON or does not match the schema.
    """
    raw = _read_json(path)
    polygon = _parse_contour(raw)
    mask = polygon_to_mask(polygon)
    contour = mask_to_contour(mask)
    return Shape(mask=mask, contour=contour, source_path=path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and JSON-decode ``path``, enforcing a top-level object."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def _parse_contour(data: dict[str, Any]) -> np.ndarray:
    """Validate the ``contour`` field and coerce it to an ``(N, 2)`` array."""
    if "contour" not in data:
        raise ValueError("missing required 'contour' key")
    raw_points = data["contour"]
    if not isinstance(raw_points, list):
        raise ValueError("'contour' must be a list of [x, y] pairs")
    if not MIN_CONTOUR_POINTS <= len(raw_points) <= MAX_CONTOUR_POINTS:
        raise ValueError(
            f"'contour' must have between {MIN_CONTOUR_POINTS} and "
            f"{MAX_CONTOUR_POINTS} points, got {len(raw_points)}"
        )
    return _coerce_points(raw_points)


def _coerce_points(raw_points: list[Any]) -> np.ndarray:
    """Convert a list of ``[x, y]`` pairs into a float32 ``(N, 2)`` array."""
    for i, pair in enumerate(raw_points):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"point {i} is not an [x, y] pair")
        if not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in pair):
            raise ValueError(f"point {i} has non-numeric coordinates")
    return np.asarray(raw_points, dtype=np.float32)

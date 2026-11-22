"""Shared synthetic-shape fixtures and geometry helpers for the test suite.

No external data is used: every shape is drawn programmatically so tests are
fast, deterministic, and self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def circle(n: int = 200, r: float = 1.0, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """A regular polygon approximating a circle."""
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def rectangle(w: float = 2.0, h: float = 1.0, n: int = 50) -> np.ndarray:
    """A rectangle perimeter sampled with ``n`` points per side."""
    xs = np.concatenate(
        [np.linspace(-w / 2, w / 2, n), np.full(n, w / 2),
         np.linspace(w / 2, -w / 2, n), np.full(n, -w / 2)]
    )
    ys = np.concatenate(
        [np.full(n, -h / 2), np.linspace(-h / 2, h / 2, n),
         np.full(n, h / 2), np.linspace(h / 2, -h / 2, n)]
    )
    return np.column_stack([xs, ys])


def triangle(n: int = 80) -> np.ndarray:
    """An asymmetric triangle perimeter (good for mirror tests)."""
    verts = np.array([[0.0, 1.0], [-0.4, -1.0], [1.0, -0.7]])
    edges = [np.linspace(verts[i], verts[(i + 1) % 3], n, endpoint=False) for i in range(3)]
    return np.vstack(edges)


def rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate points about the origin."""
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return points @ np.array([[c, -s], [s, c]]).T


def scale(points: np.ndarray, factor: float) -> np.ndarray:
    """Uniformly scale points."""
    return points * factor


def stretch(points: np.ndarray, ratio: float) -> np.ndarray:
    """Anisotropically scale along the y-axis."""
    out = points.copy()
    out[:, 1] *= ratio
    return out


def mirror(points: np.ndarray) -> np.ndarray:
    """Mirror points across the y-axis."""
    out = points.copy()
    out[:, 0] *= -1
    return out


def write_polygon(path: Path, points: np.ndarray) -> Path:
    """Write points to a one-polygon JSON file and return the path."""
    path.write_text(json.dumps({"contour": np.asarray(points).tolist()}), encoding="utf-8")
    return path


@pytest.fixture
def shapes() -> dict[str, np.ndarray]:
    """A small library of distinct base shapes."""
    return {"circle": circle(), "rectangle": rectangle(), "triangle": triangle()}


@pytest.fixture
def candidate_dir(tmp_path: Path, shapes: dict[str, np.ndarray]) -> Path:
    """A directory of candidate polygon JSON files, one per base shape."""
    out = tmp_path / "candidates"
    out.mkdir()
    for name, pts in shapes.items():
        write_polygon(out / f"{name}.json", pts)
    return out

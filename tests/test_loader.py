"""Loader tests: schema validation, rejection of bad input, round-trip fidelity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import circle, write_polygon
from shapematch.loader import load_polygon


def test_loads_valid_polygon(tmp_path: Path) -> None:
    path = write_polygon(tmp_path / "c.json", circle())
    shape = load_polygon(path)
    assert shape.mask.ndim == 2
    assert shape.mask.max() == 255
    assert shape.contour.shape[1] == 2
    assert shape.source_path == path


def test_roundtrip_preserves_shape(tmp_path: Path) -> None:
    # A circle rasterized then re-extracted should fill a similar area.
    path = write_polygon(tmp_path / "c.json", circle(r=1.0))
    shape = load_polygon(path)
    filled_fraction = float(shape.mask.mean()) / 255.0
    # A circle inscribed in its bounding box fills ~pi/4 of the padded canvas.
    assert 0.5 < filled_fraction < 0.75


def test_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_polygon(path)


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_polygon(path)


def test_rejects_missing_contour(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"points": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'contour'"):
        load_polygon(path)


def test_rejects_too_few_points(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"contour": [[0, 0], [1, 1]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="between"):
        load_polygon(path)


def test_rejects_non_numeric_point(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"contour": [[0, 0], [1, 1], ["x", 2]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        load_polygon(path)


def test_rejects_bool_coordinates(tmp_path: Path) -> None:
    # bool is a subclass of int; it must still be rejected as non-numeric.
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"contour": [[0, 0], [1, 1], [True, 2]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        load_polygon(path)


def test_rejects_malformed_pair(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"contour": [[0, 0], [1, 1], [2, 2, 2]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an \\[x, y\\] pair"):
        load_polygon(path)


def test_rejects_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_polygon(tmp_path / "does_not_exist.json")


def test_contour_is_finite(tmp_path: Path) -> None:
    path = write_polygon(tmp_path / "c.json", circle())
    shape = load_polygon(path)
    assert np.all(np.isfinite(shape.contour))

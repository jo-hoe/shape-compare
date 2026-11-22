"""Prepare Natural Earth landmasses as one-polygon candidate JSON files.

This is an **optional** helper, not part of the core tool. It turns public-domain
Natural Earth vector data into the app's native single-polygon format
(``{"contour": [[x, y], ...]}``) -- one file per country/island -- which
``shape-compare`` then consumes from a directory.

It is deliberately standalone (no console entry point) and imports the ``[prep]``
extras lazily, so the core package never depends on GIS libraries. Install them
with::

    pip install -e .[prep]

Usage::

    python tools/prep_candidates.py NATURAL_EARTH.shp data/candidates \\
        --min-area-km2 500

Each exploded polygon's exterior ring is written verbatim in an equal-area
projection -- no rasterization, so no re-tracing loss. Files are named
``<ISO_A3>__<index>.json`` (falling back to a running counter when a code is
missing), so a country split into islands yields one candidate per landmass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geopandas import GeoDataFrame
    from shapely.geometry.base import BaseGeometry

# World Equal Area (Mollweide); metres, so areas are comparable across latitudes.
_EQUAL_AREA_CRS = "ESRI:54009"
_KM2_PER_M2 = 1e-6
_MIN_RING_POINTS = 3


@dataclass(frozen=True)
class PrepConfig:
    """Resolved inputs for a single preparation run."""

    source: Path
    out_dir: Path
    min_area_km2: float


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and write candidate JSON files. Returns an exit code."""
    args = _build_parser().parse_args(argv)
    config = PrepConfig(
        source=Path(args.source),
        out_dir=Path(args.out_dir),
        min_area_km2=args.min_area_km2,
    )
    if not config.source.is_file():
        print(f"error: source not found: {config.source}", file=sys.stderr)
        return 2
    written = prepare(config)
    print(f"wrote {written} candidate polygons to {config.out_dir}")
    return 0 if written else 1


def prepare(config: PrepConfig) -> int:
    """Read, reproject, explode, filter, and write candidates. Returns the count."""
    frame = _load_equal_area(config.source)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, row in enumerate(_iter_landmasses(frame, config.min_area_km2)):
        _write_candidate(config.out_dir, row.code, index, row.ring)
        written += 1
    return written


@dataclass(frozen=True)
class _Landmass:
    """One exploded polygon ready to serialize."""

    code: str
    ring: list[list[float]]


def _load_equal_area(source: Path) -> GeoDataFrame:
    """Load the source vector file reprojected to an equal-area CRS."""
    import geopandas as gpd

    frame = gpd.read_file(source)
    return frame.to_crs(_EQUAL_AREA_CRS)


def _iter_landmasses(frame: GeoDataFrame, min_area_km2: float) -> list[_Landmass]:
    """Explode multipart geometries and keep rings above the area floor."""
    exploded = frame.explode(index_parts=False, ignore_index=True)
    landmasses: list[_Landmass] = []
    for _, row in exploded.iterrows():
        ring = _exterior_ring(row.geometry)
        if ring is None or _area_km2(row.geometry) < min_area_km2:
            continue
        landmasses.append(_Landmass(code=_iso_code(row), ring=ring))
    return landmasses


def _exterior_ring(geometry: BaseGeometry | None) -> list[list[float]] | None:
    """Return a polygon's exterior ring as ``[[x, y], ...]``, or ``None``."""
    if geometry is None or geometry.geom_type != "Polygon":
        return None
    coords = [[float(x), float(y)] for x, y in geometry.exterior.coords]
    return coords if len(coords) >= _MIN_RING_POINTS else None


def _area_km2(geometry: BaseGeometry) -> float:
    """Area of an equal-area geometry in square kilometres."""
    return float(geometry.area) * _KM2_PER_M2


def _iso_code(row: object) -> str:
    """Best-effort ISO A3 code for naming, defaulting to ``XXX``."""
    for field in ("ISO_A3", "ADM0_A3", "SOV_A3"):
        value = getattr(row, field, None)
        if isinstance(value, str) and value and value != "-99":
            return value
    return "XXX"


def _write_candidate(out_dir: Path, code: str, index: int, ring: list[list[float]]) -> None:
    """Write one ``{"contour": ring}`` JSON file."""
    path = out_dir / f"{code}__{index:04d}.json"
    path.write_text(json.dumps({"contour": ring}), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    p = argparse.ArgumentParser(
        prog="prep_candidates",
        description="Natural Earth -> one-polygon candidate JSON files.",
    )
    p.add_argument("source", help="Natural Earth vector file (.shp/.gpkg/.geojson)")
    p.add_argument("out_dir", help="directory to write candidate JSON files into")
    p.add_argument(
        "--min-area-km2",
        type=float,
        default=500.0,
        help="drop landmasses smaller than this (default: 500)",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(main())

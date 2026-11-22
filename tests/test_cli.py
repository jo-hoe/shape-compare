"""CLI tests: argument parsing, output formats, and exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import circle, rotate, triangle, write_polygon
from shapematch.cli import (
    _params_from_args,
    _parse_aspect_ratios,
    _parse_weights,
    main,
)


def _make_query(tmp_path: Path) -> tuple[Path, Path]:
    """Write a target and a candidate directory; return (target, dir)."""
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    write_polygon(candidates / "triangle.json", triangle())
    write_polygon(candidates / "circle.json", circle())
    target = write_polygon(tmp_path / "target.json", rotate(triangle(), 35.0))
    return target, candidates


def test_parse_aspect_ratios_valid() -> None:
    assert _parse_aspect_ratios("0.8,1.0,1.25") == (0.8, 1.0, 1.25)


def test_parse_aspect_ratios_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        _parse_aspect_ratios("1.0,0.0")


def test_parse_weights_defaults_to_efd_primary() -> None:
    assert _parse_weights("efd=0.9,hu=0.1") == (0.9, 0.1)


def test_parse_weights_rejects_bare_token() -> None:
    with pytest.raises(ValueError, match="key=value"):
        _parse_weights("efd")


def test_params_from_args_round_trips_flags() -> None:
    from shapematch.cli import _build_parser

    args = _build_parser().parse_args(
        ["t.json", "cands", "--mirror", "--harmonics", "12", "--top-k", "3"]
    )
    params = _params_from_args(args)
    assert params.allow_mirror is True
    assert params.harmonics == 12
    assert params.top_k == 3


def test_main_table_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target, candidates = _make_query(tmp_path)
    code = main([str(target), str(candidates), "--workers", "1", "--no-cache"])
    out = capsys.readouterr().out
    assert code == 0
    assert "triangle.json" in out
    assert "rank" in out


def test_main_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target, candidates = _make_query(tmp_path)
    code = main(
        [str(target), str(candidates), "--workers", "1", "--no-cache", "--format", "json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload[0]["candidate"] == "triangle.json"
    assert 0.0 <= payload[0]["score"] <= 1.0


def test_main_writes_json_file(tmp_path: Path) -> None:
    target, candidates = _make_query(tmp_path)
    out_file = tmp_path / "results.json"
    code = main(
        [str(target), str(candidates), "--workers", "1", "--no-cache", "--json", str(out_file)]
    )
    assert code == 0
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload[0]["candidate"] == "triangle.json"


def test_main_missing_target_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, candidates = _make_query(tmp_path)
    code = main([str(tmp_path / "nope.json"), str(candidates)])
    assert code == 2
    assert "target not found" in capsys.readouterr().err


def test_main_missing_candidates_dir_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = write_polygon(tmp_path / "t.json", triangle())
    code = main([str(target), str(tmp_path / "missing")])
    assert code == 2
    assert "candidates directory not found" in capsys.readouterr().err


def test_main_empty_candidates_dir_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = write_polygon(tmp_path / "t.json", triangle())
    empty = tmp_path / "empty"
    empty.mkdir()
    code = main([str(target), str(empty)])
    assert code == 2
    assert "no *.json candidates" in capsys.readouterr().err


def test_main_rebuild_cache_clears_stale(tmp_path: Path) -> None:
    target, candidates = _make_query(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale = cache_dir / "stale.npz"
    stale.write_bytes(b"junk")
    code = main(
        [str(target), str(candidates), "--workers", "1", "--cache-dir", str(cache_dir),
         "--rebuild-cache"]
    )
    assert code == 0
    assert not stale.exists()

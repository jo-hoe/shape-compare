"""Command-line entry point for ``shape-compare``.

Usage::

    shape-compare TARGET.json CANDIDATES_DIR [options]

The target and candidates all use the single JSON polygon format. Output is a
ranked table by default, or JSON with ``--format json`` / ``--json PATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .cache import DescriptorCache
from .config import MatchParams
from .engine import RankingEngine
from .loader import load_polygon
from .parallel import ParallelExecutor
from .types import MatchResult

_DEFAULT_CACHE_SUBDIR = ".cache"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the ranking, and print results. Returns exit code."""
    args = _build_parser().parse_args(argv)
    try:
        results = _run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(results, args)
    return 0 if results else 1


@dataclass(frozen=True)
class _RunContext:
    """Resolved inputs for a single ranking run."""

    target: Path
    candidates: list[Path]
    params: MatchParams
    cache: DescriptorCache | None
    workers: int | None


def _run(args: argparse.Namespace) -> list[MatchResult]:
    """Build the engine from parsed args and rank the candidates."""
    ctx = _context_from_args(args)
    engine = RankingEngine(
        params=ctx.params,
        executor=ParallelExecutor(workers=ctx.workers),
        cache=ctx.cache,
    )
    return engine.rank(load_polygon(ctx.target), ctx.candidates)


def _context_from_args(args: argparse.Namespace) -> _RunContext:
    """Validate paths and assemble a :class:`_RunContext`."""
    target = Path(args.target)
    if not target.is_file():
        raise FileNotFoundError(f"target not found: {target}")
    candidates_dir = Path(args.candidates)
    if not candidates_dir.is_dir():
        raise FileNotFoundError(f"candidates directory not found: {candidates_dir}")
    candidates = _discover_candidates(candidates_dir, recursive=args.recursive)
    if not candidates:
        raise ValueError(f"no *.json candidates found in {candidates_dir}")
    return _RunContext(
        target=target,
        candidates=candidates,
        params=_params_from_args(args),
        cache=_cache_from_args(args, candidates_dir),
        workers=args.workers,
    )


def _discover_candidates(directory: Path, *, recursive: bool) -> list[Path]:
    """Return sorted ``*.json`` files under ``directory``."""
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(directory.glob(pattern))


def _params_from_args(args: argparse.Namespace) -> MatchParams:
    """Construct :class:`MatchParams` from parsed CLI arguments."""
    w_efd, w_hu = _parse_weights(args.weights)
    return MatchParams(
        harmonics=args.harmonics,
        n_points=args.points,
        aspect_ratios=_parse_aspect_ratios(args.aspect_ratios),
        allow_mirror=args.mirror,
        use_shape_context=args.shape_context,
        sc_rotation_steps=args.sc_steps,
        w_efd=w_efd,
        w_hu=w_hu,
        top_k=args.top_k,
        stage1_keep=args.stage1_keep,
    )


def _cache_from_args(args: argparse.Namespace, candidates_dir: Path) -> DescriptorCache | None:
    """Resolve the descriptor cache, or ``None`` when disabled."""
    if args.no_cache:
        return None
    cache_dir = Path(args.cache_dir) if args.cache_dir else candidates_dir / _DEFAULT_CACHE_SUBDIR
    cache = DescriptorCache(cache_dir)
    if args.rebuild_cache:
        for stale in cache.directory.glob("*.npz"):
            stale.unlink()
    return cache


def _parse_aspect_ratios(text: str) -> tuple[float, ...]:
    """Parse a comma-separated list of positive aspect ratios."""
    ratios = tuple(float(part) for part in text.split(","))
    if not ratios or any(r <= 0 for r in ratios):
        raise ValueError("aspect ratios must be positive")
    return ratios


def _parse_weights(text: str) -> tuple[float, float]:
    """Parse ``efd=..,hu=..`` into ``(w_efd, w_hu)``."""
    values = dict(_split_kv(part) for part in text.split(","))
    return float(values.get("efd", 0.9)), float(values.get("hu", 0.1))


def _split_kv(part: str) -> tuple[str, str]:
    """Split ``key=value`` into a tuple, validating the shape."""
    key, _, value = part.partition("=")
    if not value:
        raise ValueError(f"expected key=value, got {part!r}")
    return key.strip(), value.strip()


def _emit(results: list[MatchResult], args: argparse.Namespace) -> None:
    """Write results to stdout and optionally a JSON file."""
    if args.json:
        Path(args.json).write_text(_to_json(results), encoding="utf-8")
    if args.format == "json":
        print(_to_json(results))
    else:
        print(_to_table(results))


def _to_json(results: list[MatchResult]) -> str:
    """Serialize results as a JSON array."""
    return json.dumps([_result_dict(r) for r in results], indent=2)


def _result_dict(result: MatchResult) -> dict[str, object]:
    """Convert a :class:`MatchResult` into a JSON-friendly dict."""
    return {
        "candidate": result.candidate.name,
        "score": round(result.score, 6),
        "efd_dist": round(result.efd_dist, 6),
        "hu_dist": round(result.hu_dist, 6),
        "sc_dist": None if result.sc_dist is None else round(result.sc_dist, 6),
        "mirrored": result.mirrored,
        "best_aspect": result.best_aspect,
    }


def _to_table(results: list[MatchResult]) -> str:
    """Render results as an aligned fixed-width table."""
    header = f"{'rank':>4}  {'candidate':<28}  {'score':>8}  {'mir':>3}  {'aspect':>6}"
    lines = [header, "-" * len(header)]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"{i:>4}  {r.candidate.name[:28]:<28}  {r.score:>8.4f}  "
            f"{'yes' if r.mirrored else 'no':>3}  {r.best_aspect:>6.2f}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    p = argparse.ArgumentParser(prog="shape-compare", description=__doc__)
    p.add_argument("target", help="target polygon JSON file")
    p.add_argument("candidates", help="directory of candidate polygon JSON files")
    p.add_argument("--mirror", action="store_true", help="allow mirrored matches")
    p.add_argument("--harmonics", type=int, default=20)
    p.add_argument("--points", type=int, default=128)
    p.add_argument("--aspect-ratios", default="0.8,0.9,1.0,1.1,1.25")
    p.add_argument("--shape-context", action="store_true", help="enable shape-context re-rank")
    p.add_argument("--sc-steps", type=int, default=36)
    p.add_argument("--stage1-keep", type=int, default=50)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--workers", type=int, default=None, help="process count (1 = serial)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--weights", default="efd=0.9,hu=0.1")
    p.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    p.add_argument("--format", choices=("table", "json"), default="table")
    p.add_argument("--json", default=None, help="also write results to this JSON file")
    return p


if __name__ == "__main__":
    raise SystemExit(main())

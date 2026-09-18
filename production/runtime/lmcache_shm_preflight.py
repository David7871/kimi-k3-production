#!/usr/bin/env python3
"""Guard LMCache POSIX SHM pools before starting a model owner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
from typing import NamedTuple

POOL_PREFIX = "lmcache_l1_pool_"


class CleanupResult(NamedTuple):
    kept: tuple[Path, ...]
    kept_requested: tuple[Path, ...]
    removed: tuple[Path, ...]
    skipped: tuple[Path, ...]


def _normalize_pool_name(name: str) -> str:
    normalized = name.lstrip("/")
    if not normalized.startswith(POOL_PREFIX):
        normalized = f"{POOL_PREFIX}{normalized}"
    return normalized


def _mapped_paths(proc_root: Path) -> set[Path]:
    mapped: set[Path] = set()
    for maps_path in proc_root.glob("[0-9]*/maps"):
        try:
            lines = maps_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in lines:
            fields = line.split(maxsplit=5)
            if len(fields) < 6:
                continue
            raw_path = fields[5]
            if raw_path.endswith(" (deleted)"):
                raw_path = raw_path[: -len(" (deleted)")]
            if raw_path.startswith("/"):
                mapped.add(Path(raw_path))
    return mapped


def cleanup_unmapped_pools(
    *,
    shm_dir: Path,
    proc_root: Path = Path("/proc"),
    exact_name: str | None = None,
    keep_name: str | None = None,
    dry_run: bool = False,
) -> CleanupResult:
    """Remove only regular LMCache pool files not mapped by any visible PID.

    keep_name pools are always kept: a same-generation pool is reusable
    across restarts, so bouncing the owning container must not drop L1.
    """
    mapped = _mapped_paths(proc_root)
    candidates = sorted(shm_dir.glob(f"{POOL_PREFIX}*"))
    if exact_name is not None:
        normalized = _normalize_pool_name(exact_name)
        candidates = [p for p in candidates if p.name == normalized]
    normalized_keep = _normalize_pool_name(keep_name) if keep_name else None

    kept: list[Path] = []
    kept_requested: list[Path] = []
    removed: list[Path] = []
    skipped: list[Path] = []
    for path in candidates:
        if normalized_keep is not None and path.name == normalized_keep:
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            kept_requested.append(path)
            continue
        try:
            before = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(before.st_mode):
            skipped.append(path)
            continue
        if path in mapped:
            kept.append(path)
            continue
        if not dry_run:
            try:
                current = path.lstat()
            except FileNotFoundError:
                continue
            if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
                skipped.append(path)
                continue
            path.unlink()
        removed.append(path)

    return CleanupResult(
        tuple(kept), tuple(kept_requested), tuple(removed), tuple(skipped)
    )


def require_capacity(
    *, free_bytes: int, required_bytes: int, reserve_bytes: int
) -> None:
    minimum = required_bytes + reserve_bytes
    if free_bytes < minimum:
        raise RuntimeError(
            "Insufficient /dev/shm capacity for required direct SHM: "
            f"free={free_bytes} required={required_bytes} "
            f"reserve={reserve_bytes} minimum={minimum}. "
            "Refusing unsafe pickle fallback."
        )


def _filesystem_free_bytes(path: Path) -> int:
    fs = os.statvfs(path)
    return fs.f_bavail * fs.f_frsize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--exact-name")
    parser.add_argument("--keep-name")
    parser.add_argument("--required-gib", type=float, default=0.0)
    parser.add_argument("--reserve-gib", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = cleanup_unmapped_pools(
        shm_dir=args.shm_dir,
        proc_root=args.proc_root,
        exact_name=args.exact_name,
        keep_name=args.keep_name,
        dry_run=args.dry_run,
    )
    for path in result.kept:
        print(f"KEEP mapped pool: {path}")
    for path in result.kept_requested:
        print(f"KEEP requested pool: {path}")
    for path in result.removed:
        action = "WOULD_REMOVE" if args.dry_run else "REMOVED"
        print(f"{action} unmapped pool: {path}")
    for path in result.skipped:
        print(f"SKIP unsafe candidate: {path}")
    if args.required_gib > 0:
        free_bytes = _filesystem_free_bytes(args.shm_dir)
        required_bytes = int(args.required_gib * 2**30)
        reserve_bytes = int(args.reserve_gib * 2**30)
        # A kept pool is capacity already allocated; only its shortfall
        # against the requirement needs fresh room.
        for path in result.kept_requested:
            try:
                required_bytes = max(0, required_bytes - path.lstat().st_size)
            except FileNotFoundError:
                pass
        require_capacity(
            free_bytes=free_bytes,
            required_bytes=required_bytes,
            reserve_bytes=reserve_bytes,
        )
        print(
            "CAPACITY_OK "
            f"free={free_bytes} required={required_bytes} reserve={reserve_bytes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

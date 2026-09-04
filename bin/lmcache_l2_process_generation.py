#!/usr/bin/env python3
"""Start one LMCache L2 process generation from an empty namespace.

LMCache MP cross-process reuse can return valid-looking but non-equivalent
hybrid recurrent state. This preflight keeps the mounted namespace itself but
removes its prior process generation before the LMCache server starts. Symlinks
are unlinked and never followed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


def inventory(path: Path) -> tuple[int, int]:
    files = 0
    size = 0
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    files += 1
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    files += 1
                    size += entry.stat(follow_symlinks=False).st_size
    return files, size


def reset(path: Path, expected_leaf: str) -> tuple[int, int]:
    if path.is_symlink():
        raise SystemExit(f"refusing symlink namespace: {path}")
    resolved = path.resolve(strict=True)
    if resolved.name != expected_leaf:
        raise SystemExit(
            f"leaf mismatch: expected {expected_leaf!r}, got {resolved.name!r}"
        )
    if resolved == Path("/") or len(resolved.parts) < 3:
        raise SystemExit(f"refusing unsafe namespace path: {resolved}")

    files, size = inventory(resolved)
    for child in resolved.iterdir():
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)
    return files, size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--expected-leaf", required=True)
    args = parser.parse_args()
    files, size = reset(args.path, args.expected_leaf)
    print(
        "LMCache L2 process-generation reset: "
        f"path={args.path} files={files} bytes={size}"
    )


if __name__ == "__main__":
    main()

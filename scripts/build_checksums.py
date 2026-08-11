#!/usr/bin/env python3
"""Regenerate the model inventory and repository SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = ROOT / "models" / "MANIFEST.csv"
REPOSITORY_MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".cache", ".matplotlib"}
EXCLUDED_FILES = {"Makefile", "uv.lock"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != REPOSITORY_MANIFEST
        and relative.as_posix() not in EXCLUDED_FILES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and path.suffix != ".pyc"
    )


def main() -> None:
    model_paths = sorted(path for path in (ROOT / "models").glob("*/*") if path.is_file())
    with MODEL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["year", "file", "bytes", "sha256"])
        writer.writeheader()
        for path in model_paths:
            writer.writerow(
                {
                    "year": path.parent.name,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )

    paths = sorted(path for path in ROOT.rglob("*") if included(path))
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths]
    REPOSITORY_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(model_paths)} model rows and {len(paths)} repository checksums")


if __name__ == "__main__":
    main()

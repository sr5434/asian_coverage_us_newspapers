#!/usr/bin/env python3
"""Remove only reproducible generated outputs, preserving all frozen inputs."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
targets = [
    ROOT / "data" / "processed",
    ROOT / "results" / "generated",
    ROOT / ".cache",
    ROOT / ".matplotlib",
]
for target in targets:
    if target.exists():
        shutil.rmtree(target)
        print(f"Removed {target.relative_to(ROOT)}")

for target in sorted(ROOT.rglob("__pycache__"), reverse=True):
    shutil.rmtree(target)
    print(f"Removed {target.relative_to(ROOT)}")

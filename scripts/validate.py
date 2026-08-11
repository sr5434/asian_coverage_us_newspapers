#!/usr/bin/env python3
"""Run all automated integrity and paper-result checks."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framing_shifts.validation import validate_all


def main() -> None:
    results = validate_all(ROOT)
    for result in results:
        print(f"[{result.status}] {result.name}: {result.evidence}")
    failures = [result for result in results if result.status == "FAIL"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Run the complete fast reproduction from included intermediate files."""

from __future__ import annotations

from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framing_shifts import change_points, stability
from framing_shifts.embedding_aggregation import aggregate_all
from framing_shifts.figures import make_all_figures
from framing_shifts.prepare import prepare_all
from framing_shifts.tables import make_all_tables


def main() -> None:
    started = time.perf_counter()
    prepared = prepare_all(ROOT)
    print(f"Prepared {len(prepared)} tidy inputs")
    margins = aggregate_all(ROOT)
    print(f"Recomputed {len(margins)} positivity-margin series")
    tables = make_all_tables(ROOT)
    print(f"Built {len(tables)} direct summary tables")
    figures = make_all_figures(ROOT)
    print(f"Rendered {len(figures)} numbered figures")
    change_points.main()
    stability.main()
    print(f"Fast reproduction completed in {time.perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()


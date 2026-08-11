#!/usr/bin/env python3
"""Regenerate BERT similarity tables from the bundled models and verify them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framing_shifts.compute_similarities import compute_year
from framing_shifts.paths import load_config


def compare_tables(generated: Path, reference: Path) -> float:
    left = pd.read_csv(generated)
    right = pd.read_csv(reference)
    key_columns = ["Word", "Sentiment"]
    if not left[key_columns].equals(right[key_columns]):
        raise AssertionError(f"Key/order mismatch for {reference.name}")
    value_columns = [column for column in right if column.endswith(" Similarity")]
    return float(np.max(np.abs(left[value_columns] - right[value_columns])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        help="Subset of years to replay (default: every configured year)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Maximum accepted CPU/GPU floating-point difference",
    )
    args = parser.parse_args()

    config = load_config(ROOT / "config" / "paper.toml")
    years = args.years or [int(year) for year in config["years"]]
    inventory = pd.read_csv(ROOT / "data" / "processed" / "adjective_inventory.csv")
    output_dir = ROOT / "results" / "generated" / "model_replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []

    for year in years:
        generated = compute_year(
            year,
            model_root=ROOT / "models",
            inventory=inventory,
            group_terms=list(config["embeddings"]["group_terms"]),
            output_dir=output_dir,
        )
        reference = (
            ROOT
            / "data"
            / "intermediate"
            / "bert_similarity_scores"
            / f"bert_similarity_stats_{year}.csv"
        )
        maximum_difference = compare_tables(generated, reference)
        status = "PASS" if maximum_difference <= args.tolerance else "FAIL"
        rows.append(
            {
                "year": year,
                "status": status,
                "rows": len(pd.read_csv(generated)),
                "maximum_absolute_difference": maximum_difference,
            }
        )
        print(f"[{status}] {year}: max abs difference={maximum_difference:.3e}")

    report = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "tolerance": args.tolerance,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device_note": "Small differences are expected between the original GPU export and CPU replay.",
        "years": rows,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "PASS":
        raise SystemExit(1)
    print(
        f"Model replay passed for {len(rows)}/{len(rows)} years in "
        f"{report['elapsed_seconds']:.2f} seconds"
    )


if __name__ == "__main__":
    main()

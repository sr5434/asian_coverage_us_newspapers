"""Aggregate frozen BERT similarity observations into paper figure inputs."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .paths import REPO_ROOT, load_config


def extract_year(path: Path) -> int:
    match = re.search(r"(\d{4})", path.stem)
    if not match:
        raise ValueError(f"No four-digit year in {path.name}")
    return int(match.group(1))


def bootstrap_mean_ci(
    values: pd.Series,
    rng: np.random.Generator,
    iterations: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.size == 1:
        return mean, mean, mean
    indices = rng.integers(0, clean.size, size=(iterations, clean.size))
    means = clean[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return mean, float(lower), float(upper)


def bootstrap_difference_ci(
    primary: pd.Series,
    baseline: pd.Series,
    rng: np.random.Generator,
    iterations: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    primary_values = primary.dropna().to_numpy(dtype=float)
    baseline_values = baseline.dropna().to_numpy(dtype=float)
    if primary_values.size == 0 or baseline_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(primary_values.mean() - baseline_values.mean())
    primary_indices = rng.integers(
        0, primary_values.size, size=(iterations, primary_values.size)
    )
    baseline_indices = rng.integers(
        0, baseline_values.size, size=(iterations, baseline_values.size)
    )
    bootstrap_differences = (
        primary_values[primary_indices].mean(axis=1)
        - baseline_values[baseline_indices].mean(axis=1)
    )
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_differences, [alpha, 1.0 - alpha])
    return mean, float(lower), float(upper)


def aggregate_similarity_scores(
    input_dir: Path,
    output_dir: Path,
    *,
    iterations: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 1910,
) -> dict[str, Path]:
    csv_paths = sorted(input_dir.glob("*.csv"), key=extract_year)
    if not csv_paths:
        raise ValueError(f"No BERT similarity CSVs found in {input_dir}")

    rng = np.random.default_rng(seed)
    expected_inventory: pd.DataFrame | None = None
    demonym_columns: list[str] | None = None
    margin_rows: dict[str, list[dict[str, float | int]]] = {}

    for csv_path in csv_paths:
        year = extract_year(csv_path)
        frame = pd.read_csv(csv_path)
        required = {"Word", "Sentiment"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{csv_path} lacks Word/Sentiment columns")
        inventory = frame[["Word", "Sentiment"]].reset_index(drop=True)
        if expected_inventory is None:
            expected_inventory = inventory
        elif not inventory.equals(expected_inventory):
            raise ValueError(f"Adjective inventory differs in {csv_path.name}")

        if demonym_columns is None:
            demonym_columns = [
                column
                for column in frame.columns
                if column.lower().endswith(" similarity") and column.lower() != "word"
            ]
        if not demonym_columns:
            raise ValueError(f"No similarity columns found in {csv_path}")

        sentiments = list(frame["Sentiment"].dropna().unique())
        sentiment_frames = {
            sentiment: frame[frame["Sentiment"] == sentiment] for sentiment in sentiments
        }

        # These calls intentionally preserve the random-number consumption order
        # of the original analysis. The resulting positivity-margin intervals are
        # therefore byte-for-byte reproducible with the frozen reference files.
        for sentiment in sentiments:
            for column in demonym_columns:
                bootstrap_mean_ci(
                    sentiment_frames[sentiment][column],
                    rng,
                    iterations,
                    confidence_level,
                )

        if "Positive" not in sentiment_frames or "Negative" not in sentiment_frames:
            raise ValueError(f"Positive and negative rows are required in {csv_path.name}")
        for column in demonym_columns:
            mean, lower, upper = bootstrap_difference_ci(
                sentiment_frames["Positive"][column],
                sentiment_frames["Negative"][column],
                rng,
                iterations,
                confidence_level,
            )
            margin_rows.setdefault(column, []).append(
                {
                    "Year": year,
                    "Positivity Margin": mean,
                    "Positivity Margin CI Lower": lower,
                    "Positivity Margin CI Upper": upper,
                }
            )

        if "Neutral" in sentiment_frames:
            for sentiment in ("Positive", "Negative"):
                if sentiment not in sentiment_frames:
                    continue
                for column in demonym_columns:
                    bootstrap_difference_ci(
                        sentiment_frames[sentiment][column],
                        sentiment_frames["Neutral"][column],
                        rng,
                        iterations,
                        confidence_level,
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for column, rows in margin_rows.items():
        label = column.removesuffix(" Similarity").casefold()
        path = output_dir / f"{label}_positivity_margin.csv"
        pd.DataFrame(rows).sort_values("Year").to_csv(path, index=False)
        outputs[label] = path
    return outputs


def aggregate_all(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    config = load_config(repo_root / "config" / "paper.toml")["embeddings"]
    return aggregate_similarity_scores(
        repo_root / "data" / "intermediate" / "bert_similarity_scores",
        repo_root / "data" / "processed" / "positivity_margins",
        iterations=int(config["bootstrap_iterations"]),
        confidence_level=float(config["confidence_level"]),
        seed=int(config["bootstrap_seed"]),
    )


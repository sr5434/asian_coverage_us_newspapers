"""Recompute every change point reported in the paper from exported chart data.

Annual series use binary segmentation with three descriptive breakpoints.
Event-level series use PELT with a single scale-independent penalty.  A returned
breakpoint at row ``b`` is reported as the boundary between rows ``b - 1`` and
``b``.  Effect size is the difference between the fitted means of the adjacent
segments, matching the piecewise-constant objective used by both algorithms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

from .paths import REPO_ROOT


DATA_DIR = REPO_ROOT / "data" / "processed"
EVENT_DATA_DIR = DATA_DIR / "events"
MARGIN_DATA_DIR = DATA_DIR / "positivity_margins"
OUTPUT_DIR = REPO_ROOT / "results" / "generated" / "tables" / "change_points"

ANNUAL_BREAKPOINTS = 3
MIN_SEGMENT_SIZE = 2
PELT_PENALTY_MULTIPLIER = 2.0


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = values.std(ddof=0)
    if scale == 0:
        return values - values.mean()
    return (values - values.mean()) / scale


def boundary_rows(
    *,
    analysis: str,
    series: str,
    labels: list[str],
    values: np.ndarray,
    breakpoints: list[int],
    algorithm: str,
    display_scale: float,
    unit: str,
) -> list[dict[str, object]]:
    starts = [0, *breakpoints[:-1]]
    segment_means = [float(values[start:end].mean()) for start, end in zip(starts, breakpoints)]
    rows: list[dict[str, object]] = []

    for segment_index, boundary_index in enumerate(breakpoints[:-1]):
        before_start = starts[segment_index]
        before_end = boundary_index
        after_start = boundary_index
        after_end = breakpoints[segment_index + 1]
        before_mean = segment_means[segment_index] * display_scale
        after_mean = segment_means[segment_index + 1] * display_scale
        immediate_before = float(values[boundary_index - 1]) * display_scale
        immediate_after = float(values[boundary_index]) * display_scale
        rows.append(
            {
                "analysis": analysis,
                "series": series,
                "algorithm": algorithm,
                "boundary_before": labels[boundary_index - 1],
                "boundary_after": labels[boundary_index],
                "before_segment_start": labels[before_start],
                "before_segment_end": labels[before_end - 1],
                "after_segment_start": labels[after_start],
                "after_segment_end": labels[after_end - 1],
                "segment_mean_before": before_mean,
                "segment_mean_after": after_mean,
                "segment_mean_change": after_mean - before_mean,
                "immediate_value_before": immediate_before,
                "immediate_value_after": immediate_after,
                "immediate_change": immediate_after - immediate_before,
                "unit": unit,
            }
        )
    return rows


def annual_coverage_rows() -> list[dict[str, object]]:
    data = pd.read_csv(DATA_DIR / "annual_name_rates.csv").sort_values("Year").reset_index(drop=True)
    labels = data["Year"].astype(int).astype(str).tolist()
    rows: list[dict[str, object]] = []
    for group in ("Chinese", "Japanese"):
        values = data[f"{group} Percentage"].to_numpy(dtype=float)
        breakpoints = rpt.Binseg(model="l2", jump=1, min_size=MIN_SEGMENT_SIZE).fit(values).predict(
            n_bkps=ANNUAL_BREAKPOINTS
        )
        rows.extend(
            boundary_rows(
                analysis="Annual coverage",
                series=f"{group} surname-match rate",
                labels=labels,
                values=values,
                breakpoints=breakpoints,
                algorithm="Binary segmentation (l2, jump=1, min_size=2, n_bkps=3)",
                display_scale=100.0,
                unit="percentage points",
            )
        )
    return rows


def sentiment_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for demonym in ("asian", "chinese", "japanese", "korean", "oriental"):
        data = pd.read_csv(MARGIN_DATA_DIR / f"{demonym}_positivity_margin.csv").sort_values("Year").reset_index(drop=True)
        labels = data["Year"].astype(int).astype(str).tolist()
        values = data["Positivity Margin"].to_numpy(dtype=float)
        breakpoints = rpt.Binseg(model="l2", jump=1, min_size=MIN_SEGMENT_SIZE).fit(values).predict(
            n_bkps=ANNUAL_BREAKPOINTS
        )
        rows.extend(
            boundary_rows(
                analysis="Annual positivity margin",
                series=demonym.capitalize(),
                labels=labels,
                values=values,
                breakpoints=breakpoints,
                algorithm="Binary segmentation (l2, jump=1, min_size=2, n_bkps=3)",
                display_scale=1.0,
                unit="positivity-margin units",
            )
        )
    return rows


def event_coverage_rows() -> list[dict[str, object]]:
    event_specs = (
        ("Opium War", "Opium_War.csv", "Chinese"),
        ("California Gold Rush", "California_Gold_Rush.csv", "Chinese"),
        ("Chinese Exclusion Act", "Asian_Exclusion_Act.csv", "Chinese"),
        ("Pearl Harbor", "Pearl_Harbor.csv", "Japanese"),
    )
    rows: list[dict[str, object]] = []

    for event, filename, group in event_specs:
        data = pd.read_csv(EVENT_DATA_DIR / filename)
        data["Date"] = pd.to_datetime(data["Month/Year"], format="%B %Y")
        data = data.sort_values("Date").reset_index(drop=True)
        labels = data["Date"].dt.strftime("%b. %Y").tolist()
        values = (data[group] / data["All"]).to_numpy(dtype=float)
        penalty = PELT_PENALTY_MULTIPLIER * np.log(len(values))
        breakpoints = rpt.Pelt(model="l2", jump=1, min_size=MIN_SEGMENT_SIZE).fit(
            standardized(values)
        ).predict(pen=penalty)
        event_rows = boundary_rows(
            analysis=event,
            series=f"{group} surname-match rate",
            labels=labels,
            values=values,
            breakpoints=breakpoints,
            algorithm=f"PELT (l2, standardized, jump=1, min_size=2, penalty={penalty:.6f})",
            display_scale=100.0,
            unit="percentage points",
        )
        if event_rows:
            rows.extend(event_rows)
        else:
            rows.append(
                {
                    "analysis": event,
                    "series": f"{group} surname-match rate",
                    "algorithm": f"PELT (l2, standardized, jump=1, min_size=2, penalty={penalty:.6f})",
                    "boundary_before": "None",
                    "boundary_after": "None",
                    "before_segment_start": labels[0],
                    "before_segment_end": labels[-1],
                    "after_segment_start": "",
                    "after_segment_end": "",
                    "segment_mean_before": float(values.mean()) * 100.0,
                    "segment_mean_after": np.nan,
                    "segment_mean_change": np.nan,
                    "immediate_value_before": np.nan,
                    "immediate_value_after": np.nan,
                    "immediate_change": np.nan,
                    "unit": "percentage points",
                }
            )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "annual_coverage.csv": annual_coverage_rows(),
        "event_coverage.csv": event_coverage_rows(),
        "sentiment.csv": sentiment_rows(),
    }
    for filename, rows in outputs.items():
        output_path = OUTPUT_DIR / filename
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Wrote {len(rows)} rows to {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

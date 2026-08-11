"""Run stability checks for every change point reported in ``paper.tex``.

The script treats the settings in ``rerun_change_points.py`` as the baseline.
It then varies defensible modeling choices and reports how often each baseline
boundary is recovered at the same location or within one observation.

Outputs are written to ``change_point_stability_results/``:

* ``specification_runs.csv``: one row per fitted specification and boundary.
* ``leave_one_out_runs.csv``: annual-series leave-one-year-out results.
* ``boundary_stability.csv``: recovery rates for each paper boundary.
* ``report.md``: concise, human-readable findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

from .paths import REPO_ROOT


DATA_DIR = REPO_ROOT / "data" / "processed"
EVENT_DATA_DIR = DATA_DIR / "events"
MARGIN_DATA_DIR = DATA_DIR / "positivity_margins"
OUTPUT_DIR = REPO_ROOT / "results" / "generated" / "tables" / "stability"

ANNUAL_MIN_SIZES = (2, 3, 4)
ANNUAL_BREAK_COUNTS = (1, 2, 3, 4)
ANNUAL_ALGORITHMS = ("Binseg", "Dynp")
EVENT_MIN_SIZES = (2, 3, 4, 6)
EVENT_PENALTY_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5, 3.0)
EVENT_ALGORITHMS = ("Pelt", "Binseg", "Dynp")
MATCH_TOLERANCE = 1


@dataclass(frozen=True)
class Series:
    analysis: str
    series: str
    labels: tuple[str, ...]
    values: np.ndarray
    kind: str


def standardized(values: np.ndarray) -> np.ndarray:
    scale = values.std(ddof=0)
    return values - values.mean() if scale == 0 else (values - values.mean()) / scale


def annual_series() -> list[Series]:
    coverage = pd.read_csv(DATA_DIR / "annual_name_rates.csv").sort_values("Year")
    labels = tuple(coverage["Year"].astype(int).astype(str))
    result = [
        Series(
            "Annual coverage",
            f"{group} surname-match rate",
            labels,
            coverage[f"{group} Percentage"].to_numpy(float),
            "annual",
        )
        for group in ("Chinese", "Japanese")
    ]
    for demonym in ("asian", "chinese", "japanese", "korean", "oriental"):
        data = pd.read_csv(MARGIN_DATA_DIR / f"{demonym}_positivity_margin.csv").sort_values("Year")
        result.append(
            Series(
                "Annual positivity margin",
                demonym.capitalize(),
                tuple(data["Year"].astype(int).astype(str)),
                data["Positivity Margin"].to_numpy(float),
                "annual",
            )
        )
    return result


def event_series() -> list[Series]:
    specs = (
        ("Opium War", "Opium_War.csv", "Chinese"),
        ("California Gold Rush", "California_Gold_Rush.csv", "Chinese"),
        ("Chinese Exclusion Act", "Asian_Exclusion_Act.csv", "Chinese"),
        ("Pearl Harbor", "Pearl_Harbor.csv", "Japanese"),
    )
    result = []
    for event, filename, group in specs:
        data = pd.read_csv(EVENT_DATA_DIR / filename)
        data["Date"] = pd.to_datetime(data["Month/Year"], format="%B %Y")
        data = data.sort_values("Date")
        result.append(
            Series(
                event,
                f"{group} surname-match rate",
                tuple(data["Date"].dt.strftime("%Y-%m")),
                (data[group] / data["All"]).to_numpy(float),
                "event",
            )
        )
    return result


def fit_fixed_count(values: np.ndarray, algorithm: str, min_size: int, count: int) -> list[int]:
    if (count + 1) * min_size > len(values):
        return []
    model = getattr(rpt, algorithm)(model="l2", jump=1, min_size=min_size).fit(values)
    return model.predict(n_bkps=count)[:-1]


def baseline_boundaries(item: Series) -> list[int]:
    if item.kind == "annual":
        return fit_fixed_count(item.values, "Binseg", 2, 3)
    penalty = 2.0 * np.log(len(item.values))
    return rpt.Pelt(model="l2", jump=1, min_size=2).fit(standardized(item.values)).predict(pen=penalty)[:-1]


def boundary_label(item: Series, boundary: int) -> str:
    return f"{item.labels[boundary - 1]}--{item.labels[boundary]}"


def specification_runs(items: list[Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in items:
        if item.kind == "annual":
            for algorithm in ANNUAL_ALGORITHMS:
                for min_size in ANNUAL_MIN_SIZES:
                    for count in ANNUAL_BREAK_COUNTS:
                        boundaries = fit_fixed_count(item.values, algorithm, min_size, count)
                        if not boundaries and (count + 1) * min_size > len(item.values):
                            continue
                        spec = f"{algorithm}; min_size={min_size}; n_bkps={count}"
                        for boundary in boundaries:
                            rows.append(
                                {
                                    "analysis": item.analysis,
                                    "series": item.series,
                                    "test_family": "annual specification grid",
                                    "specification": spec,
                                    "boundary_index": boundary,
                                    "boundary": boundary_label(item, boundary),
                                }
                            )
        else:
            signal = standardized(item.values)
            baseline_count = len(baseline_boundaries(item))
            for min_size in EVENT_MIN_SIZES:
                for multiplier in EVENT_PENALTY_MULTIPLIERS:
                    penalty = multiplier * np.log(len(signal))
                    boundaries = rpt.Pelt(model="l2", jump=1, min_size=min_size).fit(signal).predict(pen=penalty)[:-1]
                    spec = f"Pelt; min_size={min_size}; penalty={multiplier:g}*log(n)"
                    for boundary in boundaries or [None]:
                        rows.append(
                            {
                                "analysis": item.analysis,
                                "series": item.series,
                                "test_family": "event penalty/min-size grid",
                                "specification": spec,
                                "boundary_index": boundary,
                                "boundary": boundary_label(item, boundary) if boundary is not None else "None",
                            }
                        )
            if baseline_count:
                for algorithm in EVENT_ALGORITHMS[1:]:
                    for min_size in EVENT_MIN_SIZES:
                        boundaries = fit_fixed_count(signal, algorithm, min_size, baseline_count)
                        if not boundaries and (baseline_count + 1) * min_size > len(signal):
                            continue
                        spec = f"{algorithm}; min_size={min_size}; n_bkps={baseline_count}"
                        for boundary in boundaries:
                            rows.append(
                                {
                                    "analysis": item.analysis,
                                    "series": item.series,
                                    "test_family": "event algorithm grid",
                                    "specification": spec,
                                    "boundary_index": boundary,
                                    "boundary": boundary_label(item, boundary),
                                }
                            )
    return pd.DataFrame(rows)


def leave_one_out_runs(items: list[Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in items:
        if item.kind != "annual":
            continue
        full_positions = {label: index for index, label in enumerate(item.labels)}
        for omitted in range(len(item.values)):
            values = np.delete(item.values, omitted)
            labels = tuple(label for index, label in enumerate(item.labels) if index != omitted)
            boundaries = fit_fixed_count(values, "Binseg", 2, 3)
            for boundary in boundaries:
                before, after = labels[boundary - 1], labels[boundary]
                # Map a reduced-series boundary back to the full-series index of
                # its right-hand observation, preserving the paper convention.
                full_index = full_positions[after]
                rows.append(
                    {
                        "analysis": item.analysis,
                        "series": item.series,
                        "omitted": item.labels[omitted],
                        "boundary_index": full_index,
                        "boundary": f"{before}--{after}",
                    }
                )
    return pd.DataFrame(rows)


def unique_specs(frame: pd.DataFrame, analysis: str, series: str, family: str) -> list[str]:
    selected = frame[
        (frame["analysis"] == analysis)
        & (frame["series"] == series)
        & (frame["test_family"] == family)
    ]
    return sorted(selected["specification"].unique())


def summarize(items: list[Series], specs: pd.DataFrame, loo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in items:
        baselines: list[int | None] = baseline_boundaries(item)
        if item.kind == "event" and not baselines:
            baselines = [None]
        for baseline in baselines:
            families = (
                ("specification", "annual specification grid")
                if item.kind == "annual"
                else ("penalty_min_size", "event penalty/min-size grid")
            )
            output: dict[str, object] = {
                "analysis": item.analysis,
                "series": item.series,
                "baseline_boundary_index": baseline,
                "baseline_boundary": boundary_label(item, baseline) if baseline is not None else "None",
            }
            family_key, family = families
            spec_names = unique_specs(specs, item.analysis, item.series, family)
            matched = 0
            exact = 0
            for spec in spec_names:
                candidate = specs[
                    (specs["analysis"] == item.analysis)
                    & (specs["series"] == item.series)
                    & (specs["test_family"] == family)
                    & (specs["specification"] == spec)
                ]["boundary_index"].dropna().to_numpy()
                if baseline is None:
                    matched += int(len(candidate) == 0)
                    exact += int(len(candidate) == 0)
                else:
                    matched += int(np.any(np.abs(candidate - baseline) <= MATCH_TOLERANCE))
                    exact += int(np.any(candidate == baseline))
            output[f"{family_key}_runs"] = len(spec_names)
            output[f"{family_key}_recovered_within_1"] = matched
            output[f"{family_key}_recovery_rate"] = matched / len(spec_names) if spec_names else np.nan
            output[f"{family_key}_exact_rate"] = exact / len(spec_names) if spec_names else np.nan

            if baseline is None:
                output["no_breakpoint_recovery_rate"] = output[f"{family_key}_recovery_rate"]
                rows.append(output)
                continue

            if item.kind == "event":
                algorithm_names = unique_specs(specs, item.analysis, item.series, "event algorithm grid")
                algorithm_matched = 0
                for spec in algorithm_names:
                    candidate = specs[
                        (specs["analysis"] == item.analysis)
                        & (specs["series"] == item.series)
                        & (specs["test_family"] == "event algorithm grid")
                        & (specs["specification"] == spec)
                    ]["boundary_index"].to_numpy()
                    algorithm_matched += int(np.any(np.abs(candidate - baseline) <= MATCH_TOLERANCE))
                output["algorithm_runs"] = len(algorithm_names)
                output["algorithm_recovered_within_1"] = algorithm_matched
                output["algorithm_recovery_rate"] = (
                    algorithm_matched / len(algorithm_names) if algorithm_names else np.nan
                )
            else:
                selected = loo[(loo["analysis"] == item.analysis) & (loo["series"] == item.series)]
                omitted_labels = sorted(selected["omitted"].unique(), key=lambda x: item.labels.index(x))
                loo_matched = 0
                for omitted in omitted_labels:
                    candidate = selected[selected["omitted"] == omitted]["boundary_index"].to_numpy()
                    loo_matched += int(np.any(np.abs(candidate - baseline) <= MATCH_TOLERANCE))
                output["leave_one_out_runs"] = len(omitted_labels)
                output["leave_one_out_recovered_within_1"] = loo_matched
                output["leave_one_out_recovery_rate"] = loo_matched / len(omitted_labels)
                endpoint_labels = (item.labels[baseline - 1], item.labels[baseline])
                endpoint_matched = 0
                for omitted in endpoint_labels:
                    candidate = selected[selected["omitted"] == omitted]["boundary_index"].to_numpy()
                    endpoint_matched += int(np.any(np.abs(candidate - baseline) <= MATCH_TOLERANCE))
                output["endpoint_omission_runs"] = 2
                output["endpoint_omission_recovered_within_1"] = endpoint_matched
                output["endpoint_omission_recovery_rate"] = endpoint_matched / 2
            rows.append(output)
    return pd.DataFrame(rows)


def stability_label(row: pd.Series) -> str:
    rates = [row.get("specification_recovery_rate"), row.get("penalty_min_size_recovery_rate")]
    rates += [row.get("algorithm_recovery_rate"), row.get("leave_one_out_recovery_rate")]
    rates += [row.get("endpoint_omission_recovery_rate")]
    rates += [row.get("no_breakpoint_recovery_rate")]
    rates = [float(rate) for rate in rates if pd.notna(rate)]
    minimum = min(rates)
    if minimum >= 0.8:
        return "stable"
    if minimum >= 0.5:
        return "partially stable"
    return "unstable/exploratory"


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Change-point stability report",
        "",
        "Recovery means a fitted boundary occurred at the baseline location or within one observation. "
        "Annual specification tests vary algorithm, minimum segment length, and requested break count; "
        "annual leave-one-out tests omit one year at a time, with the observations adjacent to each "
        "boundary summarized separately. Event tests vary PELT penalty and minimum "
        "segment length, then compare fixed-count binary segmentation and dynamic programming.",
        "",
        "| Analysis | Series | Baseline boundary | Stability | Main recovery | Secondary recovery |",
        "|---|---|---:|---|---:|---:|",
    ]
    for _, row in summary.iterrows():
        if pd.notna(row.get("specification_recovery_rate")):
            main = f"specifications {row['specification_recovery_rate']:.0%}"
            secondary = (
                f"leave-one-out {row['leave_one_out_recovery_rate']:.0%}; "
                f"boundary endpoints {row['endpoint_omission_recovered_within_1']:.0f}/2"
            )
        elif row["baseline_boundary"] == "None":
            main = f"no-break result {row['no_breakpoint_recovery_rate']:.0%}"
            secondary = "not applicable"
        else:
            main = f"penalty/min-size {row['penalty_min_size_recovery_rate']:.0%}"
            secondary = f"algorithm {row['algorithm_recovery_rate']:.0%}"
        lines.append(
            f"| {row['analysis']} | {row['series']} | {row['baseline_boundary']} | "
            f"{row['stability']} | {main} | {secondary} |"
        )
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- Stable: every applicable recovery rate is at least 80%.",
        "- Partially stable: the lowest applicable recovery rate is 50% to 79%.",
        "- Unstable/exploratory: at least one applicable recovery rate is below 50%.",
        "",
        "These labels measure model-location stability, not data validity or historical causality. "
        "They do not replace article-level validation of unusual observations such as 1791 or 1950.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = [*annual_series(), *event_series()]
    specs = specification_runs(items)
    loo = leave_one_out_runs(items)
    summary = summarize(items, specs, loo)
    summary["stability"] = summary.apply(stability_label, axis=1)
    specs.to_csv(OUTPUT_DIR / "specification_runs.csv", index=False)
    loo.to_csv(OUTPUT_DIR / "leave_one_out_runs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "boundary_stability.csv", index=False)
    write_report(summary)
    counts = summary["stability"].value_counts().sort_index()
    detail = ", ".join(f"{label}={count}" for label, count in counts.items())
    print(f"Classified {len(summary)} boundaries ({detail})")
    print(f"\nWrote results to {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

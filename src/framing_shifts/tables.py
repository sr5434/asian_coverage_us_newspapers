"""Build paper-facing summary tables from the processed analysis inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import REPO_ROOT


def build_terminology_table(adjective_counts: pd.DataFrame) -> pd.DataFrame:
    observed = adjective_counts[adjective_counts["All"] > 0].copy()
    observed["Japanese Percentage"] = observed["Japanese"] / observed["All"] * 100
    observed["Chinese Percentage"] = observed["Chinese"] / observed["All"] * 100
    rows: list[dict[str, object]] = [
        {
            "Row Type": "average",
            "Term": "Average term",
            # This reproduces the label printed in Table 3. The calculation is
            # across all nonzero year-term rows, not only 1900; see validation report.
            "Reported Year": 1900,
            "Japanese Percentage": observed["Japanese Percentage"].mean(),
            "Chinese Percentage": observed["Chinese Percentage"].mean(),
            "Total Subjects": observed["All"].mean(),
            "Polarity": "Positive",
            "Calculation Scope": "all nonzero year-term rows",
        }
    ]
    for term, year in (("little", 1840), ("able", 1900)):
        selected = adjective_counts[
            (adjective_counts["Adjective"] == term) & (adjective_counts["Year"] == year)
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one adjective row for {term!r}, {year}; got {len(selected)}")
        row = selected.iloc[0]
        rows.append(
            {
                "Row Type": "term",
                "Term": term.capitalize(),
                "Reported Year": year,
                "Japanese Percentage": row["Japanese"] / row["All"] * 100,
                "Chinese Percentage": row["Chinese"] / row["All"] * 100,
                "Total Subjects": row["All"],
                "Polarity": str(row["Polarity"]).capitalize(),
                "Calculation Scope": "year-term row",
            }
        )
    return pd.DataFrame(rows)


def build_exclusion_act_comparison(event_data: pd.DataFrame) -> pd.DataFrame:
    data = event_data.copy()
    data["Date"] = pd.to_datetime(data["Month/Year"], format="%B %Y")
    windows = (
        ("January-April 1882", "1882-01-01", "1882-04-30"),
        ("May-August 1882", "1882-05-01", "1882-08-31"),
    )
    rows = []
    for label, start, end in windows:
        selected = data[data["Date"].between(start, end)]
        rows.append(
            {
                "Window": label,
                "Chinese Matches": int(selected["Chinese"].sum()),
                "All Detected Names": int(selected["All"].sum()),
                "Chinese Percentage": selected["Chinese"].sum() / selected["All"].sum() * 100,
            }
        )
    result = pd.DataFrame(rows)
    result["Change From Previous (pp)"] = result["Chinese Percentage"].diff()
    return result


def make_all_tables(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    processed = repo_root / "data" / "processed"
    output = repo_root / "results" / "generated" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    terminology_path = output / "table_3_terminology.csv"
    build_terminology_table(pd.read_csv(processed / "adjective_counts.csv")).to_csv(
        terminology_path, index=False
    )
    exclusion_path = output / "exclusion_act_descriptive_comparison.csv"
    build_exclusion_act_comparison(
        pd.read_csv(processed / "events" / "Asian_Exclusion_Act.csv")
    ).to_csv(exclusion_path, index=False)
    return {"table_3": terminology_path, "exclusion_act": exclusion_path}


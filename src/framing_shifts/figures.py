"""Render the ten numbered paper figures from tidy processed CSV files."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from .paths import REPO_ROOT


cache_root = Path(tempfile.gettempdir()) / "framing-shifts-plot-cache"
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns


sns.set_theme()


def save_chart(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    plt.savefig(path)
    plt.close()
    return path


def add_name_proportions(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for group in ("Chinese", "Japanese"):
        proportion = f"{group} Percentage"
        average = f"Average {group} Percentage"
        data[proportion] = data[group] / data["All"]
        data[average] = data[proportion].mean()
    return data


def format_percentage_axis() -> None:
    plt.ylabel("Surname Matches Among Detected Person Names")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


def plot_event(
    data_path: Path,
    output_dir: Path,
    output_filename: str,
    group: str,
) -> Path:
    data = pd.read_csv(data_path)
    data["Month/Year"] = pd.to_datetime(data["Month/Year"], format="%B %Y")
    data = add_name_proportions(data)
    plt.figure(figsize=(10, 6))
    format_percentage_axis()
    sns.lineplot(
        data=data,
        x="Month/Year",
        y=f"{group} Percentage",
        label=f"{group} Surnames",
    )
    sns.lineplot(
        data=data,
        x="Month/Year",
        y=f"Average {group} Percentage",
        label=f"Average {group} Surnames",
    )
    return save_chart(output_dir, output_filename)


def plot_annual_coverage(data_path: Path, output_dir: Path) -> Path:
    data = pd.read_csv(data_path)
    plt.figure(figsize=(10, 6))
    format_percentage_axis()
    sns.lineplot(
        data=data,
        x="Year",
        y="Chinese Percentage",
        label="Chinese Surname-Match Rate",
    )
    sns.lineplot(
        data=data,
        x="Year",
        y="Japanese Percentage",
        label="Japanese Surname-Match Rate",
    )
    return save_chart(output_dir, "chart.png")


def plot_positivity_margin(
    data_path: Path,
    output_dir: Path,
    output_filename: str,
    title_label: str,
) -> Path:
    data = pd.read_csv(data_path)
    plt.figure(figsize=(10, 6))
    color = sns.color_palette(n_colors=1)[0]
    sns.lineplot(
        data=data,
        x="Year",
        y="Positivity Margin",
        label="Positive - Negative Similarity",
        color=color,
    )
    plt.fill_between(
        data["Year"],
        data["Positivity Margin CI Lower"],
        data["Positivity Margin CI Upper"],
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    plt.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    plt.ylabel("Positivity Margin")
    plt.title(f'"{title_label}" positivity margin')
    return save_chart(output_dir, output_filename)


def make_all_figures(repo_root: Path = REPO_ROOT) -> dict[int, Path]:
    processed = repo_root / "data" / "processed"
    events = processed / "events"
    margins = processed / "positivity_margins"
    output = repo_root / "results" / "generated" / "figures"

    figures = {
        1: plot_annual_coverage(processed / "annual_name_rates.csv", output),
        2: plot_event(events / "Opium_War.csv", output, "Opium War.png", "Chinese"),
        3: plot_positivity_margin(
            margins / "asian_positivity_margin.csv", output, "Asian.png", "Asian"
        ),
        4: plot_positivity_margin(
            margins / "chinese_positivity_margin.csv", output, "Chinese.png", "Chinese"
        ),
        5: plot_event(
            events / "California_Gold_Rush.csv", output, "Gold Rush.png", "Chinese"
        ),
        6: plot_event(
            events / "Asian_Exclusion_Act.csv",
            output,
            "Asian Exclusion Act.png",
            "Chinese",
        ),
        7: plot_event(
            events / "Pearl_Harbor.csv", output, "Pearl Harbor.png", "Japanese"
        ),
        8: plot_positivity_margin(
            margins / "japanese_positivity_margin.csv", output, "Japanese.png", "Japanese"
        ),
        9: plot_positivity_margin(
            margins / "korean_positivity_margin.csv", output, "Korean.png", "Korean"
        ),
        10: plot_positivity_margin(
            margins / "oriental_positivity_margin.csv", output, "Oriental.png", "Oriental"
        ),
    }
    return figures

"""Build the paper's tidy analysis inputs from frozen intermediate files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import REPO_ROOT, load_config


EVENT_WINDOWS = {
    "Opium_War.csv": ("1838-01-01", "1842-12-01"),
    "California_Gold_Rush.csv": ("1847-01-01", "1851-12-01"),
    "Asian_Exclusion_Act.csv": ("1881-01-01", "1882-12-01"),
    "Pearl_Harbor.csv": ("1939-01-01", "1942-12-01"),
}


@dataclass(order=True, frozen=True)
class LexiconEntry:
    strength_rank: int
    part_of_speech_rank: int
    polarity: str


PART_OF_SPEECH_PRIORITY = {
    "adj": 0,
    "anypos": 1,
    "adv": 2,
    "verb": 3,
    "noun": 4,
}
ALLOWED_POLARITIES = {"positive", "negative", "both", "neutral"}


def read_monthly_name_counts(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        required = ["Month/Year", "Japanese", "Chinese", "All"]
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        frame = frame[required].copy()
        frame["Date"] = pd.to_datetime(frame["Month/Year"], format="%B %Y")
        frame["Source File"] = csv_path.name
        frames.append(frame)

    if not frames:
        raise ValueError(f"No monthly count CSVs found in {input_dir}")

    monthly = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    duplicates = monthly[monthly.duplicated("Date", keep=False)]
    if not duplicates.empty:
        detail = duplicates[["Month/Year", "Source File"]].to_dict("records")
        raise ValueError(f"Duplicate monthly observations: {detail}")
    if (monthly[["Japanese", "Chinese", "All"]] < 0).any().any():
        raise ValueError("Name counts must be non-negative")
    if ((monthly["Japanese"] > monthly["All"]) | (monthly["Chinese"] > monthly["All"])).any():
        raise ValueError("Surname-match counts cannot exceed all detected names")
    return monthly


def build_annual_name_rates(monthly: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    selected_years = set(int(year) for year in years)
    annual = monthly.assign(Year=monthly["Date"].dt.year)
    annual = (
        annual[annual["Year"].isin(selected_years)]
        .groupby("Year", as_index=False)[["All", "Chinese", "Japanese"]]
        .sum()
        .rename(
            columns={
                "All": "Total Names",
                "Chinese": "Chinese Names",
                "Japanese": "Japanese Names",
            }
        )
        .sort_values("Year")
        .reset_index(drop=True)
    )
    missing_years = sorted(selected_years - set(annual["Year"]))
    if missing_years:
        raise ValueError(f"Monthly inputs do not cover configured years: {missing_years}")
    annual["Chinese Percentage"] = annual["Chinese Names"] / annual["Total Names"]
    annual["Japanese Percentage"] = annual["Japanese Names"] / annual["Total Names"]
    return annual


def build_event_inputs(monthly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for filename, (start, end) in EVENT_WINDOWS.items():
        selected = monthly[monthly["Date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        output = selected[["Month/Year", "Japanese", "Chinese", "All"]].reset_index(drop=True)
        # These two constants were present in the plotting inputs used for the
        # paper. They are window-level means, repeated on every monthly row.
        output["Average Japanese"] = output["Japanese"].mean()
        output["Average Chinese"] = output["Chinese"].mean()
        outputs[filename] = output
    return outputs


def load_mpqa_lexicon(path: Path) -> dict[str, LexiconEntry]:
    lexicon: dict[str, LexiconEntry] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            polarity = fields.get("priorpolarity")
            word = fields.get("word1")
            if not word or polarity not in ALLOWED_POLARITIES:
                continue
            entry = LexiconEntry(
                strength_rank=0 if fields.get("type") == "strongsubj" else 1,
                part_of_speech_rank=PART_OF_SPEECH_PRIORITY.get(fields.get("pos1", ""), 5),
                polarity=polarity,
            )
            normalized_word = word.casefold()
            current = lexicon.get(normalized_word)
            if current is None or entry < current:
                lexicon[normalized_word] = entry
    return lexicon


def build_adjective_counts(batch_dir: Path, mpqa_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(batch_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        required = {"Year", "Adjective", "Japanese", "Chinese", "All"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        frames.append(frame[list(required)])
    if not frames:
        raise ValueError(f"No adjective count CSVs found in {batch_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["Adjective"] = combined["Adjective"].astype(str).str.strip().str.casefold()
    if (combined[["Japanese", "Chinese", "All"]] < 0).any().any():
        raise ValueError("Adjective association counts must be non-negative")

    # The original adjective JSON contained duplicate strings. Grouping preserves
    # their counts while giving the clean package one row per year and term.
    deduplicated = (
        combined.groupby(["Year", "Adjective"], as_index=False)[["Japanese", "Chinese", "All"]]
        .sum()
        .sort_values(["Year", "Adjective"])
        .reset_index(drop=True)
    )
    mpqa = load_mpqa_lexicon(mpqa_path)
    deduplicated["Polarity"] = deduplicated["Adjective"].map(
        lambda word: mpqa[word].polarity if word in mpqa else pd.NA
    )
    return deduplicated


def build_adjective_inventory(adjective_counts: pd.DataFrame) -> pd.DataFrame:
    observed = adjective_counts[adjective_counts["All"] > 0]
    totals = (
        observed[observed["Polarity"].isin(["positive", "negative", "neutral"])]
        .groupby(["Adjective", "Polarity"], as_index=False)["All"]
        .sum()
        .rename(columns={"All": "Total Mentions"})
    )

    inventories: dict[str, pd.DataFrame] = {}
    for polarity in ("positive", "negative", "neutral"):
        inventory = totals[totals["Polarity"] == polarity].sort_values(
            ["Total Mentions", "Adjective"], ascending=[False, True]
        )
        inventories[polarity] = inventory.reset_index(drop=True)

    balanced_size = min(len(inventories["positive"]), len(inventories["negative"]))
    selected_frames: list[pd.DataFrame] = []
    for polarity in ("positive", "negative", "neutral"):
        inventory = inventories[polarity]
        if polarity in {"positive", "negative"}:
            inventory = inventory.iloc[:balanced_size]
        inventory = inventory.copy()
        inventory["Rank"] = range(1, len(inventory) + 1)
        selected_frames.append(inventory)

    result = pd.concat(selected_frames, ignore_index=True)
    result["Sentiment"] = result["Polarity"].str.capitalize()
    return result[["Adjective", "Sentiment", "Total Mentions", "Rank"]]


def prepare_all(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    config = load_config(repo_root / "config" / "paper.toml")
    intermediate = repo_root / "data" / "intermediate"
    processed = repo_root / "data" / "processed"
    events_dir = processed / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    monthly = read_monthly_name_counts(intermediate / "name_counts_monthly")
    annual = build_annual_name_rates(monthly, config["years"])
    annual_path = processed / "annual_name_rates.csv"
    annual.to_csv(annual_path, index=False)

    outputs: dict[str, Path] = {"annual": annual_path}
    for filename, frame in build_event_inputs(monthly).items():
        path = events_dir / filename
        frame.to_csv(path, index=False)
        outputs[f"event:{filename}"] = path

    adjective_counts = build_adjective_counts(
        intermediate / "adjective_counts_by_batch",
        repo_root / "data" / "lexicons" / "mpqa" / "subjclueslen1-HLTEMNLP05.tff",
    )
    adjective_path = processed / "adjective_counts.csv"
    adjective_counts.to_csv(adjective_path, index=False)
    inventory_path = processed / "adjective_inventory.csv"
    build_adjective_inventory(adjective_counts).to_csv(inventory_path, index=False)
    outputs["adjective_counts"] = adjective_path
    outputs["adjective_inventory"] = inventory_path
    return outputs

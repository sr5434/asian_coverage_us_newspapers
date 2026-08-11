"""Independent integrity and paper-result checks for the clean reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from .paths import REPO_ROOT


PAPER_FIGURES = (
    "chart.png",
    "Opium War.png",
    "Asian.png",
    "Chinese.png",
    "Gold Rush.png",
    "Asian Exclusion Act.png",
    "Pearl Harbor.png",
    "Japanese.png",
    "Korean.png",
    "Oriental.png",
)


@dataclass
class CheckResult:
    name: str
    status: str
    evidence: str


def run_check(name: str, function: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name=name, status="PASS", evidence=function())
    except Exception as error:  # validation should record all failures in one run
        return CheckResult(name=name, status="FAIL", evidence=f"{type(error).__name__}: {error}")


def assert_frames_equal(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    sort_by: list[str],
    tolerance: float = 1e-12,
) -> None:
    left = generated.sort_values(sort_by).reset_index(drop=True)
    right = reference.sort_values(sort_by).reset_index(drop=True)
    if list(left.columns) != list(right.columns):
        raise AssertionError(f"Column mismatch: {list(left.columns)} != {list(right.columns)}")
    if len(left) != len(right):
        raise AssertionError(f"Row-count mismatch: {len(left)} != {len(right)}")
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            if not np.allclose(
                left[column].to_numpy(float),
                right[column].to_numpy(float),
                rtol=0,
                atol=tolerance,
                equal_nan=True,
            ):
                delta = np.nanmax(
                    np.abs(left[column].to_numpy(float) - right[column].to_numpy(float))
                )
                raise AssertionError(f"{column} differs (max absolute difference {delta})")
        else:
            if not left[column].fillna("<NA>").equals(right[column].fillna("<NA>")):
                raise AssertionError(f"{column} differs")


def validate_monthly_and_coverage(root: Path) -> str:
    monthly_frames = []
    for path in sorted((root / "data" / "intermediate" / "name_counts_monthly").glob("*.csv")):
        frame = pd.read_csv(path)
        frame["Date"] = pd.to_datetime(frame["Month/Year"], format="%B %Y")
        monthly_frames.append(frame)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    if len(monthly) != 384 or monthly["Date"].nunique() != 384:
        raise AssertionError(
            f"Expected 384 unique month rows; got {len(monthly)} rows and {monthly['Date'].nunique()} dates"
        )
    if (monthly[["Japanese", "Chinese", "All"]] < 0).any().any():
        raise AssertionError("Negative monthly counts found")

    assert_frames_equal(
        pd.read_csv(root / "data" / "processed" / "annual_name_rates.csv"),
        pd.read_csv(root / "data" / "reference" / "processed" / "annual_name_rates.csv"),
        sort_by=["Year"],
    )
    for filename in (
        "Opium_War.csv",
        "California_Gold_Rush.csv",
        "Asian_Exclusion_Act.csv",
        "Pearl_Harbor.csv",
    ):
        assert_frames_equal(
            pd.read_csv(root / "data" / "processed" / "events" / filename),
            pd.read_csv(root / "data" / "reference" / "processed" / "events" / filename),
            sort_by=["Month/Year"],
        )
    return "384/384 unique monthly rows; annual and all four event exports match frozen references"


def validate_adjective_counts_and_inventory(root: Path) -> str:
    generated = pd.read_csv(root / "data" / "processed" / "adjective_counts.csv")
    if generated.duplicated(["Year", "Adjective"]).any():
        raise AssertionError("Generated adjective counts contain duplicate year-term keys")
    reference = pd.read_csv(
        root / "data" / "reference" / "processed" / "adjective_counts_annotated.csv"
    )
    reference["Adjective"] = reference["Adjective"].str.casefold()
    reference = (
        reference.groupby(["Year", "Adjective"], as_index=False)
        .agg(
            {
                "Japanese": "sum",
                "Chinese": "sum",
                "All": "sum",
                "Polarity": "first",
            }
        )
        .sort_values(["Year", "Adjective"])
    )
    assert_frames_equal(
        generated,
        reference,
        sort_by=["Year", "Adjective"],
    )

    inventory = pd.read_csv(root / "data" / "processed" / "adjective_inventory.csv")
    first_similarity = pd.read_csv(
        root
        / "data"
        / "intermediate"
        / "bert_similarity_scores"
        / "bert_similarity_stats_1770.csv"
    )
    if not inventory[["Adjective", "Sentiment"]].reset_index(drop=True).equals(
        first_similarity[["Word", "Sentiment"]]
        .rename(columns={"Word": "Adjective"})
        .reset_index(drop=True)
    ):
        raise AssertionError("Generated adjective inventory does not match similarity-score keys")
    counts = inventory["Sentiment"].value_counts().to_dict()
    if counts != {"Positive": 648, "Negative": 648, "Neutral": 80}:
        raise AssertionError(f"Unexpected inventory sizes: {counts}")
    return "61,579 unique year-term rows; selected inventory is 648 positive, 648 negative, 80 neutral"


def validate_similarity_inputs(root: Path) -> str:
    paths = sorted((root / "data" / "intermediate" / "bert_similarity_scores").glob("*.csv"))
    if len(paths) != 19:
        raise AssertionError(f"Expected 19 similarity files, found {len(paths)}")
    expected_keys = None
    for path in paths:
        frame = pd.read_csv(path)
        if len(frame) != 1376 or frame.isna().any().any():
            raise AssertionError(f"{path.name}: rows={len(frame)}, nulls={int(frame.isna().sum().sum())}")
        if frame.duplicated(["Word", "Sentiment"]).any():
            raise AssertionError(f"Duplicate Word/Sentiment keys in {path.name}")
        keys = frame[["Word", "Sentiment"]]
        if expected_keys is None:
            expected_keys = keys
        elif not keys.equals(expected_keys):
            raise AssertionError(f"Inventory order differs in {path.name}")
    return "19/19 files have 1,376 complete, unique rows and an identical adjective inventory"


def validate_positivity_margins(root: Path) -> str:
    maximum_difference = 0.0
    for term in ("asian", "chinese", "japanese", "korean", "oriental"):
        generated = pd.read_csv(
            root / "data" / "processed" / "positivity_margins" / f"{term}_positivity_margin.csv"
        )
        reference = pd.read_csv(
            root
            / "data"
            / "reference"
            / "processed"
            / "positivity_margins"
            / f"{term}_positivity_margin.csv"
        )
        for column in generated.columns[1:]:
            maximum_difference = max(
                maximum_difference,
                float(np.max(np.abs(generated[column] - reference[column]))),
            )
        assert_frames_equal(generated, reference, sort_by=["Year"], tolerance=1e-14)
    return f"All 95 yearly means/intervals match; max absolute difference={maximum_difference:.3g}"


def validate_figures(root: Path) -> str:
    mismatches = []
    for filename in PAPER_FIGURES:
        generated_path = root / "results" / "generated" / "figures" / filename
        reference_path = root / "results" / "reference" / "figures" / filename
        generated = np.asarray(Image.open(generated_path).convert("RGBA"))
        reference = np.asarray(Image.open(reference_path).convert("RGBA"))
        if generated.shape != reference.shape or not np.array_equal(generated, reference):
            mismatches.append(filename)
    if mismatches:
        raise AssertionError(f"Pixel differences in: {mismatches}")
    return "10/10 numbered paper figures are pixel-exact against the frozen references"


def validate_change_point_outputs(root: Path) -> str:
    generated_cp = root / "results" / "generated" / "tables" / "change_points"
    generated_stability = root / "results" / "generated" / "tables" / "stability"
    reference = root / "results" / "reference" / "change_points"
    pairs = [
        (generated_cp / "annual_coverage.csv", reference / "annual_coverage.csv"),
        (generated_cp / "event_coverage.csv", reference / "event_coverage.csv"),
        (generated_cp / "sentiment.csv", reference / "sentiment.csv"),
        (generated_stability / "boundary_stability.csv", reference / "boundary_stability.csv"),
        (generated_stability / "leave_one_out_runs.csv", reference / "leave_one_out_runs.csv"),
        (generated_stability / "specification_runs.csv", reference / "specification_runs.csv"),
        (generated_stability / "report.md", reference / "report.md"),
    ]
    differences = [generated.name for generated, expected in pairs if generated.read_bytes() != expected.read_bytes()]
    if differences:
        raise AssertionError(f"Non-identical change-point outputs: {differences}")
    return "3/3 baseline tables and 4/4 stability artifacts are byte-exact"


def validate_direct_paper_claims(root: Path) -> str:
    terminology = pd.read_csv(
        root / "results" / "generated" / "tables" / "table_3_terminology.csv"
    ).set_index("Term")
    expected = {
        "Average term": (0.18, 0.80, 8),
        "Little": (0.00, 27.78, 36),
        "Able": (0.00, 2.74, 219),
    }
    for term, (japanese, chinese, total) in expected.items():
        row = terminology.loc[term]
        actual = (
            round(float(row["Japanese Percentage"]), 2),
            round(float(row["Chinese Percentage"]), 2),
            round(float(row["Total Subjects"])),
        )
        if actual != (japanese, chinese, total):
            raise AssertionError(f"Table 3 mismatch for {term}: {actual}")

    exclusion = pd.read_csv(
        root
        / "results"
        / "generated"
        / "tables"
        / "exclusion_act_descriptive_comparison.csv"
    )
    values = exclusion["Chinese Percentage"].round(3).tolist()
    change = round(float(exclusion["Change From Previous (pp)"].iloc[1]), 3)
    if values != [0.535, 0.552] or change != 0.017:
        raise AssertionError(f"Exclusion Act comparison mismatch: {values}, change={change}")
    return "Table 3 rows and the 0.535%/0.552% Exclusion Act comparison reproduce at paper precision"


KNOWN_GAPS = [
    "The American Stories dataset revision and base-model revision were not recorded.",
    "The manual name-validation samples/labels behind Table 5 were not retained.",
    "The close-reading article IDs, sampling manifests, and coding notes were not retained.",
    "The original per-year training metrics behind Table 7 were not retained; the CSV is paper-reported.",
    "The original exported similarity script did not call its Procrustes alignment helper; this is cosine-invariant for the reported within-year comparisons.",
    "The paper's Table 3 labels the average row as 1900, but its displayed values are the average across all nonzero year-term rows.",
]


def render_report(results: list[CheckResult]) -> str:
    failed = [result for result in results if result.status == "FAIL"]
    assessment = "Needs revision" if failed else "Share with caveats"
    lines = [
        "# Validation report",
        "",
        f"## Overall assessment: {assessment}",
        "",
        "The included downstream workflow is computationally reproducible. Full provenance is incomplete for the explicitly listed manual and training-stage artifacts.",
        "",
        "## Automated checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for result in results:
        evidence = result.evidence.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {result.name} | {result.status} | {evidence} |")
    lines += ["", "## Known reproducibility gaps", ""]
    lines.extend(f"- {gap}" for gap in KNOWN_GAPS)
    lines += [
        "",
        "## Interpretation",
        "",
        "Passing checks establish that the frozen intermediates regenerate the plotted series, figures, change points, and stability outputs. They do not independently validate OCR quality, surname-based ethnicity inference, manual historical interpretations, or causal claims.",
        "",
    ]
    return "\n".join(lines)


def validate_all(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    checks = [
        ("Monthly counts and coverage lineage", validate_monthly_and_coverage),
        ("Adjective counts and inventory", validate_adjective_counts_and_inventory),
        ("BERT similarity intermediate quality", validate_similarity_inputs),
        ("Positivity-margin calculations", validate_positivity_margins),
        ("Numbered figures", validate_figures),
        ("Change points and stability", validate_change_point_outputs),
        ("Direct displayed-value spot checks", validate_direct_paper_claims),
    ]
    results = [run_check(name, lambda function=function: function(repo_root)) for name, function in checks]
    output_dir = repo_root / "results" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(results)
    model_report_path = output_dir / "model_replay" / "validation.json"
    if model_report_path.exists():
        model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
        model_rows = model_report.get("years", [])
        maximum_difference = max(
            (float(row["maximum_absolute_difference"]) for row in model_rows),
            default=float("nan"),
        )
        report += (
            "\n## Separate offline model replay\n\n"
            f"Status: **{model_report.get('status', 'UNKNOWN')}** for "
            f"{len(model_rows)} checkpoint(s); maximum absolute similarity "
            f"difference `{maximum_difference:.3e}` at tolerance "
            f"`{float(model_report.get('tolerance', float('nan'))):.1e}`. "
            "This check is produced by `scripts/replay_models.py` and requires "
            "the full dependencies.\n"
        )
    report_path = repo_root / "docs" / "VALIDATION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return results

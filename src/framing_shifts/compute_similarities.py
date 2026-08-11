"""Compute adjective/group-term similarities from locally trained BERT models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from .bert_embeddings import BertEmbeddingModel, align_to_anchor
from .paths import REPO_ROOT, load_config


def parse_years(values: Iterable[str]) -> list[int]:
    years: list[int] = []
    for value in values:
        years.extend(int(item) for item in value.split(",") if item.strip())
    return years


def compute_year(
    year: int,
    *,
    model_root: Path,
    inventory: pd.DataFrame,
    group_terms: list[str],
    output_dir: Path,
    anchor: BertEmbeddingModel | None = None,
) -> Path:
    model = BertEmbeddingModel.from_directory(year, model_root / str(year))
    if anchor is not None and year != anchor.year:
        align_to_anchor(model, anchor)

    rows: list[dict[str, object]] = []
    for item in inventory.itertuples(index=False):
        adjective = item.Adjective
        if adjective not in model:
            continue
        similarities = {
            term: model.similarity(adjective, term) if term in model else None
            for term in group_terms
        }
        row: dict[str, object] = {"Word": adjective, "Sentiment": item.Sentiment}
        row.update({f"{term.capitalize()} Similarity": value for term, value in similarities.items()})
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"bert_similarity_stats_{year}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", required=True, help="Years, space- or comma-separated")
    parser.add_argument("--model-root", type=Path, default=REPO_ROOT / "models")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "adjective_inventory.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw_generated" / "bert_similarity_scores",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Apply the paper-described 1910 Procrustes rotation. It is cosine-invariant here.",
    )
    args = parser.parse_args()
    config = load_config()
    years = parse_years(args.years)
    inventory = pd.read_csv(args.inventory)
    embedding_config = config["embeddings"]
    anchor = None
    if args.align:
        anchor_year = int(embedding_config["anchor_year"])
        anchor = BertEmbeddingModel.from_directory(anchor_year, args.model_root / str(anchor_year))
    for year in years:
        output = compute_year(
            year,
            model_root=args.model_root,
            inventory=inventory,
            group_terms=list(embedding_config["group_terms"]),
            output_dir=args.output_dir,
            anchor=anchor,
        )
        print(f"{year}: wrote {output}")


if __name__ == "__main__":
    main()


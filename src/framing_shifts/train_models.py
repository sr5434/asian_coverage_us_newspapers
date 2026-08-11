"""Fine-tune one local BERT-Tiny masked-language model per newspaper year."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

from .paths import REPO_ROOT, load_config


def parse_years(values: Iterable[str]) -> list[int]:
    years: list[int] = []
    for value in values:
        years.extend(int(item) for item in value.split(",") if item.strip())
    return years


def update_metrics(path: Path, row: dict[str, object]) -> None:
    existing: dict[int, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for current in csv.DictReader(handle):
                existing[int(current["year"])] = current
    existing[int(row["year"])] = {key: str(value) for key, value in row.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for year in sorted(existing):
            writer.writerow(existing[year])


def train_year(year: int, model_root: Path, metrics_path: Path) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    config = load_config()
    dataset_config = config["dataset"]
    sampling = config["sampling"]
    training = config["training"]
    base_model = training["base_model"]

    model_kwargs = {}
    if training["base_model_revision"] != "UNRECORDED":
        model_kwargs["revision"] = training["base_model_revision"]
    model = AutoModelForMaskedLM.from_pretrained(base_model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model, **model_kwargs)
    dataset_kwargs = {
        "path": dataset_config["identifier"],
        "name": dataset_config["configuration"],
        "year_list": [str(year)],
        "trust_remote_code": True,
    }
    if dataset_config["revision"] != "UNRECORDED":
        dataset_kwargs["revision"] = dataset_config["revision"]
    dataset = load_dataset(**dataset_kwargs)[str(year)]
    max_articles = int(sampling["embedding_analysis_max_articles"])
    selected = dataset.shuffle(seed=int(sampling["embedding_analysis_seed"])).select(
        range(min(len(dataset), max_articles))
    )

    def tokenize_function(examples):
        result = tokenizer(examples["article"])
        if tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(index) for index in range(len(result["input_ids"]))
            ]
        return result

    tokenized = selected.map(
        tokenize_function,
        batched=True,
        remove_columns=selected.column_names,
        desc=f"Tokenizing {year}",
    )
    chunk_size = int(training["sequence_length"])

    def group_texts(examples):
        concatenated = {key: sum(examples[key], []) for key in examples}
        total_length = len(concatenated[next(iter(concatenated))])
        total_length = (total_length // chunk_size) * chunk_size
        grouped = {
            key: [values[index : index + chunk_size] for index in range(0, total_length, chunk_size)]
            for key, values in concatenated.items()
        }
        grouped["labels"] = grouped["input_ids"].copy()
        return grouped

    chunks = tokenized.map(group_texts, batched=True, desc=f"Grouping {year} into chunks")
    # This intentionally preserves the original notebook's split. For large
    # years it selects 90,000 train and 9,000 test chunks, leaving extra chunks
    # unused; it is not a split over source articles.
    train_size = int(min(0.9 * max_articles, 0.9 * len(chunks)))
    test_size = int(0.1 * train_size)
    split = chunks.train_test_split(
        train_size=train_size,
        test_size=test_size,
        seed=int(training["trainer_seed"]),
    )

    batch_size = int(training["batch_size"])
    work_dir = model_root / "work" / str(year)
    args = TrainingArguments(
        output_dir=str(work_dir),
        learning_rate=float(training["learning_rate"]),
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=float(training["epochs"]),
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        logging_steps=max(1, len(split["train"]) // batch_size),
        seed=int(training["trainer_seed"]),
        data_seed=int(training["trainer_seed"]),
        push_to_hub=False,
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=float(training["mask_probability"]),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
    )
    train_result = trainer.train()
    evaluation = trainer.evaluate()
    perplexity = math.exp(float(evaluation["eval_loss"]))

    output_dir = model_root / str(year)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    update_metrics(
        metrics_path,
        {
            "year": year,
            "perplexity": f"{perplexity:.12g}",
            "eval_loss": f"{float(evaluation['eval_loss']):.12g}",
            "train_loss": f"{float(train_result.training_loss):.12g}",
            "source_articles": len(dataset),
            "sampled_articles": len(selected),
            "token_chunks": len(chunks),
            "train_chunks": len(split["train"]),
            "test_chunks": len(split["test"]),
            "base_model": base_model,
            "trainer_seed": int(training["trainer_seed"]),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", required=True, help="Years, space- or comma-separated")
    parser.add_argument("--model-root", type=Path, default=REPO_ROOT / "models")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=REPO_ROOT / "results" / "generated" / "tables" / "model_perplexities.csv",
    )
    args = parser.parse_args()
    for year in parse_years(args.years):
        output = train_year(year, args.model_root, args.metrics)
        print(f"{year}: saved local model to {output}")


if __name__ == "__main__":
    main()

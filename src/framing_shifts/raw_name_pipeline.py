"""Regenerate name and adjective counts from American Stories.

This is the cleaned, parameterized replacement for the manually toggled cells
in ``framingshifts.ipynb``. Raw-corpus processing is expensive; the fast paper
reproduction uses the frozen intermediate outputs included in ``data/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Iterable

from .paths import REPO_ROOT, load_config


ORGANIZATION_WORDS = {
    "co", "corp", "company", "inc", "ltd", "realty", "motor", "motors",
    "chemical", "drug", "electric", "shoe", "united", "association",
    "federation", "university", "school", "church", "hospital",
}
BAD_ENTITY_WORDS = {
    "one", "only", "can", "date", "law", "man", "son", "ton", "co",
    "the", "this", "that", "mrs", "mr", "miss", "dr",
}
DANGEROUS_CHINESE_SURNAMES = {
    "lee", "ma", "mar", "mac", "la", "lou", "lay", "jo", "o", "law",
    "sun", "moon", "won", "hi", "hu", "ty", "ju", "ow", "roe", "lowe",
    "lang", "sang", "ho", "lo", "lu", "si", "sa", "ka", "ko", "ke",
    "man", "son", "ton", "can", "one", "co",
}
DANGEROUS_JAPANESE_SURNAMES = {"ike", "ito", "date", "kishi"}
PLACE_OR_NONPERSON_WORDS = {
    "highway", "road", "rd", "street", "st", "avenue", "ave", "arlington",
    "washington", "hatteras", "springhill", "locust", "west", "dominion",
    "prairie", "lodge", "sears", "fire", "moon", "friends", "executor",
    "treasurer", "sponsored",
}
EXTRA_BAD_TOKENS = {
    "county", "district", "eistrict", "community", "civic", "postmaster",
    "capt", "captain", "fort", "riverside", "desert", "chevrolet", "buick",
    "stevens", "hymn", "chain", "gang", "song", "old", "dear", "quick",
    "modern", "veto", "executor", "treasurer", "sponsored", "association",
    "highway", "road", "street", "avenue", "school",
}
OCR_JUNK_FRAGMENTS = {
    " i'd", "12-ga", "h. s.", "ker-", "wag-", "gunder-", "asso", "voca-",
    "produc", "rehabilita-", "washinston",
}
UNUSUAL_CHARACTERS = {"&", "%", "=", "~", "$", ":", ";", "_", "|", "\\", "/"}


def load_surnames(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {name.casefold() for name in json.load(handle)["romanizations"]}


def load_adjectives(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        terms = [term.casefold() for term in json.load(handle)["adjectives"]]
    # Preserve source multiplicity: the original notebook emitted one row for
    # every list entry, including duplicates. ``prepare.py`` subsequently groups
    # those rows, matching the frozen paper intermediates exactly.
    return terms


def looks_like_person_entity(entity) -> bool:
    text = entity.text.strip()
    if entity.label_ != "PERSON" or "\n" in text:
        return False
    if any(symbol in text for symbol in UNUSUAL_CHARACTERS):
        return False
    lowered_text = text.casefold()
    if any(fragment in lowered_text for fragment in OCR_JUNK_FRAGMENTS):
        return False

    alpha_tokens = [token for token in entity if token.is_alpha]
    if not 2 <= len(alpha_tokens) <= 4:
        return False
    lowered = {token.text.casefold() for token in alpha_tokens}
    if lowered & (ORGANIZATION_WORDS | BAD_ENTITY_WORDS | PLACE_OR_NONPERSON_WORDS):
        return False
    if lowered & EXTRA_BAD_TOKENS:
        return False
    if text.isupper() and len(alpha_tokens) > 1:
        return False
    if not alpha_tokens[0].text[0].isupper() or not alpha_tokens[-1].text[0].isupper():
        return False

    for token in alpha_tokens:
        value = token.text
        if len(value) == 1 or (len(value) == 2 and value.isupper()):
            continue
        if not (value.istitle() or value.isupper()):
            return False
    short_tokens = [token for token in alpha_tokens if len(token.text) <= 2]
    if len(short_tokens) >= 2 or len(short_tokens) / len(alpha_tokens) > 0.5:
        return False
    return True


def surname_for(entity) -> str:
    alpha_tokens = [token for token in entity if token.is_alpha]
    return alpha_tokens[-1].text.casefold() if len(alpha_tokens) >= 2 else ""


def associated_adjectives(entity, known_adjectives: set[str]) -> list[str]:
    matches: list[str] = []
    for token in entity:
        for child in token.children:
            if child.dep_ == "amod" and child.pos_ == "ADJ":
                lemma = child.lemma_.casefold()
                if lemma in known_adjectives:
                    matches.append(lemma)
        if token.dep_ not in {"nsubj", "nsubjpass"}:
            continue
        predicate = token.head
        if predicate.pos_ == "ADJ":
            lemma = predicate.lemma_.casefold()
            if lemma in known_adjectives:
                matches.append(lemma)
        if predicate.pos_ in {"VERB", "AUX"}:
            for child in predicate.children:
                if child.dep_ in {"acomp", "xcomp", "attr"} and child.pos_ == "ADJ":
                    lemma = child.lemma_.casefold()
                    if lemma in known_adjectives:
                        matches.append(lemma)
    return matches


def fresh_state() -> dict:
    return {
        "processed_articles": 0,
        "monthly_all": [0] * 12,
        "monthly_chinese": [0] * 12,
        "monthly_japanese": [0] * 12,
        "adjective_all": {},
        "adjective_chinese": {},
        "adjective_japanese": {},
    }


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return fresh_state()
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    template = fresh_state()
    template.update(state)
    return template


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(path)


def article_month(article: dict) -> int:
    try:
        month = int(str(article["date"]).split("-")[1])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Unparseable article date: {article.get('date')!r}") from error
    if not 1 <= month <= 12:
        raise ValueError(f"Article month outside 1..12: {article.get('date')!r}")
    return month


def process_year(
    year: int,
    *,
    output_root: Path,
    save_contexts: bool = False,
    checkpoint_interval: int = 1000,
) -> tuple[Path, Path]:
    from datasets import load_dataset
    import spacy

    config = load_config()
    dataset_config = config["dataset"]
    sampling = config["sampling"]
    lexicon_dir = REPO_ROOT / "data" / "lexicons"
    chinese_surnames = load_surnames(lexicon_dir / "chinese_names.json")
    japanese_surnames = load_surnames(lexicon_dir / "japanese_names.json")
    adjective_list = load_adjectives(lexicon_dir / "adjective_sentiment_data.json")
    adjective_set = set(adjective_list)

    nlp = spacy.load(config["ner"]["model"])
    load_kwargs = {
        "path": dataset_config["identifier"],
        "name": dataset_config["configuration"],
        "year_list": [str(year)],
        "trust_remote_code": True,
    }
    if dataset_config["revision"] != "UNRECORDED":
        load_kwargs["revision"] = dataset_config["revision"]
    dataset = load_dataset(**load_kwargs)[str(year)]
    limit = min(len(dataset), int(sampling["name_analysis_max_articles"]))
    sampled = dataset.shuffle(seed=int(sampling["name_analysis_seed"])).select(range(limit))

    checkpoint_path = output_root / "checkpoints" / f"{year}.json"
    state = load_checkpoint(checkpoint_path)
    processed = min(int(state["processed_articles"]), limit)
    adjective_all = Counter(state["adjective_all"])
    adjective_chinese = Counter(state["adjective_chinese"])
    adjective_japanese = Counter(state["adjective_japanese"])
    context_path = output_root / "contexts" / f"{year}.jsonl"
    if save_contexts:
        context_path.parent.mkdir(parents=True, exist_ok=True)

    remaining = sampled.select(range(processed, limit))
    context_handle = context_path.open("a", encoding="utf-8") if save_contexts else None
    try:
        for offset, article in enumerate(remaining, start=processed):
            month_index = article_month(article) - 1
            document = nlp(article["article"])
            for entity in document.ents:
                if not looks_like_person_entity(entity):
                    continue
                surname = surname_for(entity)
                is_chinese = (
                    surname in chinese_surnames and surname not in DANGEROUS_CHINESE_SURNAMES
                )
                is_japanese = (
                    surname in japanese_surnames and surname not in DANGEROUS_JAPANESE_SURNAMES
                )
                state["monthly_all"][month_index] += 1
                state["monthly_chinese"][month_index] += int(is_chinese)
                state["monthly_japanese"][month_index] += int(is_japanese)

                adjectives = associated_adjectives(entity, adjective_set)
                for adjective in adjectives:
                    adjective_all[adjective] += 1
                    adjective_chinese[adjective] += int(is_chinese)
                    adjective_japanese[adjective] += int(is_japanese)

                if context_handle is not None and (is_chinese or is_japanese):
                    context_handle.write(
                        json.dumps(
                            {
                                "article_id": article.get("article_id"),
                                "date": article.get("date"),
                                "newspaper_name": article.get("newspaper_name"),
                                "headline": article.get("headline"),
                                "entity": entity.text,
                                "surname": surname,
                                "chinese_match": is_chinese,
                                "japanese_match": is_japanese,
                                "associated_adjectives": adjectives,
                                "sentence": entity.sent.text.replace("\n", " ").strip(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            state["processed_articles"] = offset + 1
            if state["processed_articles"] % checkpoint_interval == 0:
                state["adjective_all"] = dict(adjective_all)
                state["adjective_chinese"] = dict(adjective_chinese)
                state["adjective_japanese"] = dict(adjective_japanese)
                save_checkpoint(checkpoint_path, state)
    finally:
        if context_handle is not None:
            context_handle.close()

    state["adjective_all"] = dict(adjective_all)
    state["adjective_chinese"] = dict(adjective_chinese)
    state["adjective_japanese"] = dict(adjective_japanese)
    state["total_articles"] = len(dataset)
    state["sampled_articles"] = limit
    state["completed"] = state["processed_articles"] == limit
    save_checkpoint(checkpoint_path, state)

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    name_output = output_root / "name_counts_monthly" / f"{year}.csv"
    name_output.parent.mkdir(parents=True, exist_ok=True)
    with name_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Month/Year", "Japanese", "Chinese", "All"])
        writer.writeheader()
        for month_index, month_name in enumerate(month_names):
            writer.writerow(
                {
                    "Month/Year": f"{month_name} {year}",
                    "Japanese": state["monthly_japanese"][month_index],
                    "Chinese": state["monthly_chinese"][month_index],
                    "All": state["monthly_all"][month_index],
                }
            )

    adjective_output = output_root / "adjective_counts" / f"{year}.csv"
    adjective_output.parent.mkdir(parents=True, exist_ok=True)
    with adjective_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Year", "Adjective", "Japanese", "Chinese", "All"]
        )
        writer.writeheader()
        for adjective in adjective_list:
            writer.writerow(
                {
                    "Year": year,
                    "Adjective": adjective,
                    "Japanese": adjective_japanese[adjective],
                    "Chinese": adjective_chinese[adjective],
                    "All": adjective_all[adjective],
                }
            )
    return name_output, adjective_output


def parse_years(values: Iterable[str]) -> list[int]:
    years: list[int] = []
    for value in values:
        years.extend(int(item) for item in value.split(",") if item.strip())
    return years


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", required=True, help="Years, space- or comma-separated")
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "data" / "raw_generated"
    )
    parser.add_argument("--save-contexts", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    args = parser.parse_args()
    for year in parse_years(args.years):
        outputs = process_year(
            year,
            output_root=args.output_root,
            save_contexts=args.save_contexts,
            checkpoint_interval=args.checkpoint_interval,
        )
        print(f"{year}: wrote {outputs[0]} and {outputs[1]}")


if __name__ == "__main__":
    main()

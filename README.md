# Anonymous reproduction package

This repository accompanies **"Surname-Based Person References and Lexical Associations in Historical American Print Media"** and can be used to reproduce the paper's results. It has been refactored and cleaned up quite a bit from the original codebase, but the results are identical.

## Quick start: figures and downstream tables

The authors used Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
python3 scripts/reproduce.py
python3 scripts/validate.py
```

Running from scratch on an M2 Mac Mini's CPU took 30ish seconds. Generated files are written under `data/processed/` and `results/generated/` and frozen comparison targets remain under `data/reference/` and `results/reference/`.

## Offline similarity recalculation

The exact 19 fine-tuned checkpoints are can be found in `models/<year>/`. The following script will recalculate the similarity scores that were used for the paper.

```bash
python3 -m pip install -r requirements-full.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/replay_models.py
```

The script automatically compares its recalculated results to the paper's values and accepts at most `1e-6` deviation by default, but this can be adjusted by the tolerance argument. The verified all-year CPU replay takes about 30 seconds on an M2 Mac Mini. 

## Raw-corpus and retraining path

The refactored raw pipeline is also available, but it is very time consuming to run.

```bash
python -m pip install -r requirements-full.txt
python -m spacy download en_core_web_md==3.8.0

PYTHONPATH=src python -m framing_shifts.raw_name_pipeline --years 1900 --save-contexts
PYTHONPATH=src python -m framing_shifts.train_models --years 1900
PYTHONPATH=src python -m framing_shifts.compute_similarities --years 1900
```

Configuration and all recorded seeds live in [config/paper.toml](config/paper.toml).

## Repository map

| Path | Purpose |
|---|---|
| `src/framing_shifts/` | Parameterized preparation, modeling, plotting, tables, and validation code |
| `data/intermediate/` | Frozen monthly counts, adjective counts, and BERT similarity observations (these get processed further by ```scripts/reproduce.py```) |
| `data/reference/` | Frozen processed outputs for comparison with regenerated data  |
| `data/reported/` | Values reported by the paper (includes data like perplexities and precision that are not in the other directories) |
| `data/lexicons/` | Lexicons used in the paper, including lists of common Chinese and Japanese surnames and data on the sentiment associated with adjectives  |
| `models/` | Fine-tuned BERT-Tiny checkpoints for all 19 years |
| `results/reference/` | Frozen figures and change-point/stability artifacts |
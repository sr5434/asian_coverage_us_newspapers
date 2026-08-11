#!/usr/bin/env python3
"""Build the clean reviewer notebook with nbformat."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "reproduce_paper.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        markdown(
            """
            # Reproducing the paper’s computational results

            This notebook is the reviewer-facing entry point for the anonymous reproduction package. It executes the deterministic downstream workflow from frozen intermediates, runs independent integrity checks, and displays representative results.
            """
        ),
        markdown(
            """
            ## TL;DR

            The retained computational pipeline reproduces all 10 numbered figures, all yearly BERT positivity-margin estimates, and every retained change-point/stability artifact. Exact BERT checkpoints are also bundled for an optional offline replay.

            Raw-corpus equality is not claimed: immutable source revisions, article manifests, manual-validation labels, close-reading samples, and saved perplexity evaluations were not retained. See `docs/LIMITATIONS.md`.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import platform
            import subprocess
            import sys

            import matplotlib
            import numpy as np
            import pandas as pd
            from IPython.display import Image, Markdown, display

            ROOT = Path.cwd().resolve()
            if not (ROOT / "config" / "paper.toml").exists():
                ROOT = ROOT.parent
            if not (ROOT / "config" / "paper.toml").exists():
                raise RuntimeError("Run this notebook from the repository root or notebooks directory")

            sys.path.insert(0, str(ROOT / "src"))
            print(
                f"Python {platform.python_version()} | NumPy {np.__version__} | "
                f"pandas {pd.__version__} | Matplotlib {matplotlib.__version__}"
            )
            """
        ),
        markdown(
            """
            ## Context and methods

            The paper measures (1) Chinese/Japanese surname matches among detected person names and (2) cosine associations between five group terms and evaluative adjectives in separate, year-specific BERT-Tiny models.

            The fast workflow starts from retained monthly count tables, adjective-association counts, and 19 BERT similarity tables. It reconstructs annual/event inputs, annotates and balances the adjective inventory, repeats the 5,000-draw bootstrap with seed 1910, renders figures, and reruns baseline/sensitivity change-point analyses.

            The reported lexical-association model is BERT-Tiny.
            """
        ),
        markdown("## Data inventory"),
        code(
            """
            monthly_paths = sorted((ROOT / "data/intermediate/name_counts_monthly").glob("*.csv"))
            monthly = pd.concat([pd.read_csv(path) for path in monthly_paths], ignore_index=True)
            similarity_paths = sorted((ROOT / "data/intermediate/bert_similarity_scores").glob("*.csv"))
            first_similarity = pd.read_csv(similarity_paths[0])

            inventory = pd.DataFrame(
                [
                    ["Monthly surname-count exports", len(monthly_paths), len(monthly)],
                    ["Per-year BERT similarity exports", len(similarity_paths), sum(len(pd.read_csv(path)) for path in similarity_paths)],
                    ["Terms per BERT export", 1, len(first_similarity)],
                    ["Fine-tuned model directories", len(list((ROOT / "models").glob("[0-9]*"))), None],
                ],
                columns=["Artifact", "Files", "Rows"],
            )
            display(inventory)
            display(first_similarity["Sentiment"].value_counts().rename("Terms").to_frame())
            """
        ),
        markdown(
            """
            ## Deterministic downstream replay

            The command below rebuilds processed data, tables, figures, baseline change points, and stability analyses. Its verbose sensitivity matrix is captured to keep this notebook readable.
            """
        ),
        code(
            """
            run = subprocess.run(
                [sys.executable, "scripts/reproduce.py"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if run.returncode:
                raise RuntimeError(run.stderr[-4000:] or run.stdout[-4000:])
            prefixes = ("Prepared", "Recomputed", "Built", "Rendered", "Fast reproduction")
            print("\\n".join(line for line in run.stdout.splitlines() if line.startswith(prefixes)))
            """
        ),
        markdown("## Validation matrix"),
        code(
            """
            validation_run = subprocess.run(
                [sys.executable, "scripts/validate.py"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if validation_run.returncode:
                raise RuntimeError(validation_run.stderr[-4000:] or validation_run.stdout[-4000:])
            print(validation_run.stdout)

            validation = pd.DataFrame(
                json.loads((ROOT / "results/generated/validation.json").read_text())
            )
            display(validation)
            """
        ),
        markdown("## Direct result tables"),
        code(
            """
            terminology = pd.read_csv(
                ROOT / "results/generated/tables/table_3_terminology.csv"
            )
            exclusion = pd.read_csv(
                ROOT / "results/generated/tables/exclusion_act_descriptive_comparison.csv"
            )
            display(Markdown("### Table 3 reconstruction"))
            display(terminology)
            display(Markdown("### Chinese Exclusion Act descriptive comparison"))
            display(exclusion)
            """
        ),
        markdown("## Representative regenerated figures"),
        code(
            """
            for number, filename in [
                (1, "chart.png"),
                (4, "Chinese.png"),
                (7, "Pearl Harbor.png"),
            ]:
                display(Markdown(f"### Figure {number}"))
                display(Image(filename=ROOT / "results/generated/figures" / filename))
            """
        ),
        markdown(
            """
            ## Optional offline model replay

            `python scripts/replay_models.py` regenerates all 19 × 1,376 similarity rows from bundled local checkpoints. It is kept outside the default notebook run to avoid repeatedly loading every model. If a saved replay report is present, its summary appears below.
            """
        ),
        code(
            """
            model_report_path = ROOT / "results/generated/model_replay/validation.json"
            if model_report_path.exists():
                model_report = json.loads(model_report_path.read_text())
                model_rows = pd.DataFrame(model_report["years"])
                print(
                    f"{model_report['status']}: {len(model_rows)}/{len(model_rows)} years; "
                    f"max abs difference={model_rows['maximum_absolute_difference'].max():.3e}; "
                    f"tolerance={model_report['tolerance']:.1e}"
                )
                display(model_rows)
            else:
                print("Model replay has not been run in this checkout; see README.md.")
            """
        ),
        markdown(
            """
            ## Takeaways

            - The frozen downstream analysis is exactly reproducible in the tested environment.
            - The bundled checkpoints independently support the BERT similarity intermediates to strict CPU/GPU tolerance.
            - The strongest remaining gaps are provenance gaps, not hidden numerical mismatches: source revisions, article manifests, manual labels, close-reading evidence, and original perplexity evaluations are absent.
            - Table 3’s printed “1900 average” is actually an average across all nonzero year-term rows; the generated table records that scope explicitly.
            - Automated checks do not validate the substantive assumptions of surname-based categorization or causal historical interpretation.
            """
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

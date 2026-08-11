from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    config_path = path or REPO_ROOT / "config" / "paper.toml"
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


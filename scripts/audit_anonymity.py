#!/usr/bin/env python3
"""Fail on common credentials, machine paths, identity metadata, or repo debris."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "", ".cfg", ".csv", ".env", ".example", ".gitignore", ".gitattributes",
    ".ipynb", ".json", ".lock", ".md", ".py", ".sh", ".tff", ".toml", ".txt",
    ".yaml", ".yml",
}
PATTERNS = {
    "token-shaped secret": re.compile(r"\b(?:hf_|sk-)[A-Za-z0-9_-]{20,}\b"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "local absolute path": re.compile(r"(?:/Users|/home|/root|/kaggle)/[^\s\"']+"),
    "private key material": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
FORBIDDEN_NAMES = {
    ".DS_Store",
    ".cache",
    ".ipynb_checkpoints",
    ".matplotlib",
    "__pycache__",
}


@dataclass
class Finding:
    kind: str
    path: str
    detail: str


def safe_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    findings: list[Finding] = []
    for path in sorted(ROOT.rglob("*")):
        relative = safe_relative(path)
        if any(part == ".git" for part in path.relative_to(ROOT).parts):
            findings.append(Finding("git metadata", relative, "fresh releases must omit history"))
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix == ".pyc":
            findings.append(Finding("generated debris", relative, path.name))
        if path.is_symlink():
            findings.append(Finding("symlink", relative, "release files must be portable"))
        if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                # Do not print the matched secret or identity-bearing value.
                line = text.count("\n", 0, match.start()) + 1
                findings.append(Finding(label, relative, f"match at line {line}"))

    report = {
        "status": "PASS" if not findings else "FAIL",
        "files_scanned": sum(1 for path in ROOT.rglob("*") if path.is_file()),
        "findings": [asdict(finding) for finding in findings],
    }
    output = ROOT / "results" / "generated" / "anonymity_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[{report['status']}] anonymity audit: {len(findings)} finding(s)")
    for finding in findings:
        print(f"- {finding.kind}: {finding.path} ({finding.detail})")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

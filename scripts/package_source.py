#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


ROOT_FILES = {
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    ".env.example",
    ".gitignore",
}
SOURCE_DIRS = {"configs", "src", "scripts", "notebooks", "tests", "docs"}
# Directory names anywhere in a path that must never enter the bundle.
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def selected_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() not in ROOT_FILES and relative.parts[0] not in SOURCE_DIRS:
            continue
        if EXCLUDED_DIRS.intersection(relative.parts):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.name.endswith(".pyc"):
            continue
        yield path, relative


def main() -> None:
    parser = argparse.ArgumentParser(description="Create organizer source bundle")
    parser.add_argument("--output", default="artifacts/alqac2026_source.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    files = list(selected_files(root))
    for path, relative in files:
        if path.suffix in {".py", ".md", ".yaml", ".toml", ".txt", ".json", ".ipynb"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"alqac_[A-Za-z0-9_-]{20,}", text, flags=re.IGNORECASE):
                raise ValueError(f"Potential hard-coded ALQAC token in {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in files:
            archive.write(path, relative)
    print(f"Created {output} with {len(files)} files")


if __name__ == "__main__":
    main()

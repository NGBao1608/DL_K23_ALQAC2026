from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def build_manifest(config: dict[str, Any], corpus_path: str | Path) -> dict[str, Any]:
    revision, dirty = git_revision()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "git_commit": revision,
        "dirty_worktree": dirty,
        "corpus_sha256": sha256_file(corpus_path),
        "models": {
            "outcome": config["prediction"]["model_name"],
            "outcome_revision": config["prediction"].get("revision"),
            "embedding": config["law_retrieval"]["embedding_model"],
            "embedding_revision": config["law_retrieval"].get("embedding_revision"),
            "reranker": config["law_retrieval"]["reranker_model"],
            "reranker_revision": config["law_retrieval"].get("reranker_revision"),
        },
        "model_licenses": {
            "outcome": "Apache-2.0",
            "embedding": "Apache-2.0",
            "reranker": "Apache-2.0",
        },
        "config": config,
    }


def build_environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "sentence-transformers",
        "bitsandbytes",
        "accelerate",
        "peft",
        "numpy",
        "pyvi",
        "rank-bm25",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = {"available": False, "device": None}
    try:
        import torch

        cuda["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cuda["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
    }


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)

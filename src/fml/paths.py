"""Where things live: the repo root, the config files and the private data repo."""
from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class DataNotFound(RuntimeError):
    """The private data repo is not where we expected it. Raised loudly instead of guessing."""


@functools.lru_cache
def project_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "project.yaml").read_text(encoding="utf-8"))


def data_dir() -> Path:
    """Return the root of the private data repo (the folder that contains loco_911/).

    Order: $FML_DATA_DIR, then the candidates in config/project.yaml (relative to the repo root),
    then any sibling folder of the repo root that contains the subset file.
    """
    cfg = project_config()["data"]
    subset = cfg["subset_file"]
    candidates: list[Path] = []
    if os.environ.get("FML_DATA_DIR"):
        candidates.append(Path(os.environ["FML_DATA_DIR"]).expanduser())
    for c in cfg["repo_dir_candidates"]:
        p = Path(c).expanduser()
        candidates.append(p if p.is_absolute() else (REPO_ROOT / p))
    candidates += [p for p in REPO_ROOT.parent.iterdir() if p.is_dir() and p != REPO_ROOT]
    for c in candidates:
        if (c / subset).is_file():
            return c.resolve()
    tried = "\n  ".join(str(c) for c in candidates[:8])
    raise DataNotFound(
        f"Could not find {subset}. Set FML_DATA_DIR to the private data repo. Tried:\n  {tried}"
    )

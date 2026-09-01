"""Shared GAMUS data-access helpers for scripts (train/validation/sampling).

Downloads h5 slices directly from the HF repo cache; keeps per-process memo so
repeated tile access is a single download. Intentionally a private helper, not
a user-facing module.
"""

from __future__ import annotations

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO = "earthflow/GAMUS"
CITIES = ["DC", "PHL", "NYC"]
IMG_SUFFIX = {"DC": "_RGB.h5", "PHL": "_RGB.h5", "NYC": "_IMG.h5"}
SPLIT = {"DC": "val", "PHL": "val", "NYC": "test"}  # held-out split (NYC has no val)

HELDOUT = [
    "DC:DC_11_16", "DC:DC_12_17", "DC:DC_13_14",
    "PHL:PHL_6150", "PHL:PHL_6151", "PHL:PHL_6152",
    "NYC:NYC_00918", "NYC:NYC_00921", "NYC:NYC_00735",
]

_memo: dict = {}


def _read(path_in_repo: str) -> np.ndarray:
    if path_in_repo in _memo:
        return _memo[path_in_repo]
    p = hf_hub_download(REPO, path_in_repo, repo_type="dataset")
    with h5py.File(p, "r") as f:
        arr = f["image"][...]
    _memo[path_in_repo] = arr
    return arr


def tile_rgb(city: str, split: str, tid: str) -> np.ndarray:
    return _read(f"images/{split}/{tid}{IMG_SUFFIX[city]}")


def tile_ndsm(city: str, split: str, tid: str) -> np.ndarray:
    return _read(f"heights/{split}/{tid}_AGL.h5").astype(np.float64)


def tile_cls(city: str, split: str, tid: str) -> np.ndarray:
    return _read(f"classes/{split}/{tid}_CLS.h5").astype(np.uint8)


def heldout_entries():
    return [(t.split(":")[0].strip(), t.split(":")[1].strip()) for t in HELDOUT]


def train_tiles(city: str, exclude: set[str] | None = None, limit: int | None = None) -> list[str]:
    """First `limit` train-split tiles for a city, excluding the held-out ids."""
    api = HfApi()
    suffix = IMG_SUFFIX[city]
    files = list(api.list_repo_tree(REPO, path_in_repo=f"images/train", repo_type="dataset", recursive=False))
    tiles = sorted(
        f.path.split("/")[-1][: -len(suffix)]
        for f in files
        if f.path.split("/")[-1].startswith(city + "_") and f.path.split("/")[-1].endswith(suffix)
    )
    exclude = exclude or set()
    tiles = [t for t in tiles if t not in exclude]
    return tiles if limit is None else tiles[:limit]
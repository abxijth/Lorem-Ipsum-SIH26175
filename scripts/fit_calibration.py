"""Fit the relative-depth -> structural-height calibration on GAMUS train.

Samples N tiles per city across the train split (all 3 cities), runs tiled
Depth-Anything depth inference on each, pools depth vs. AGL-nDSM pairs and fits
a robust global linear regression (RANSAC). Writes models/calibration.json.

Usage:
    prime-run python scripts/fit_calibration.py [--per-city 20] [--patch 518] [--stride 400]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from depthwizard.calibrate import fit_calibration  # noqa: E402
from depthwizard.config import CALIBRATION_PATH  # noqa: E402
from depthwizard.depth import DepthEstimator  # noqa: E402

REPO = "earthflow/GAMUS"
CITY_SPLITS = {"DC": "train", "PHL": "train", "NYC": "train"}
CITY_IMG_SUFFIX = {"DC": "_RGB.h5", "PHL": "_RGB.h5", "NYC": "_IMG.h5"}


def load_cached(path_in_repo, cache):
    if path_in_repo in cache:
        return cache[path_in_repo]
    p = hf_hub_download(REPO, path_in_repo, repo_type="dataset")
    with h5py.File(p, "r") as f:
        arr = f["image"][...]
    cache[path_in_repo] = arr
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=20, help="tiles sampled per city")
    ap.add_argument("--cities", default="DC,PHL,NYC", help="comma-separated city prefixes")
    ap.add_argument("--out", default=CALIBRATION_PATH, help="output calibration JSON")
    args = ap.parse_args()

    est = DepthEstimator()
    print(f"depth estimator ready ({est.device})")

    api = HfApi()
    cache = {}
    depth_samples, ndsm_samples = [], []
    city_counts = {}

    for city in args.cities.split(","):
        city = city.strip().upper()
        split, img_suffix = CITY_SPLITS[city], CITY_IMG_SUFFIX[city]
        files = list(api.list_repo_tree(REPO, path_in_repo=f"images/{split}", repo_type="dataset", recursive=False))
        tiles = sorted(f.path.split("/")[-1][: -len(img_suffix)] for f in files if f.path.split("/")[-1].startswith(city + "_"))
        tiles = tiles[: args.per_city]
        print(f"[{city}] fitting on {len(tiles)} train tiles")

        for tid in tiles:
            t0 = time.time()
            rgb = load_cached(f"images/{split}/{tid}{img_suffix}", cache)
            ndsm = load_cached(f"heights/{split}/{tid}_AGL.h5", cache).astype(float)
            depth = est.predict(rgb)
            h = min(depth.shape[0], ndsm.shape[0], rgb.shape[0])
            w = min(depth.shape[1], ndsm.shape[1], rgb.shape[1])
            depth_samples.append(depth[:h, :w])
            ndsm_samples.append(ndsm[:h, :w])
            city_counts[city] = city_counts.get(city, 0) + 1
            print(f"    {city} {tid} ({time.time()-t0:.1f}s)")

    calib = fit_calibration(depth_samples, ndsm_samples)
    calib.city_counts = city_counts
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    calib.to_json(args.out)
    print(f"\nCalibration: slope={calib.slope:.4f} intercept={calib.intercept:.2f}")
    print(f"pixels={calib.n_pixels} tiles={calib.n_tiles} cities={city_counts}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
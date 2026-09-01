"""Fit relative-depth -> structural-height calibration for the FINE-TUNED backbone.

The fine-tuned backbone outputs nDSM-like structure (metres, via NORM scale) but
with a residual scale/offset vs ground truth. We anchor it with the same robust
global linear (RANSAC) calibration the frozen baseline used, fit on GAMUS train
depth (from the fine-tuned model) vs AGL nDSM. Writes models/calibration.json.

This is the production calibration shipped alongside the fine-tuned weights.

Usage:
    prime-run python scripts/fit_calibration_finetuned.py [--per-city 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _gamus  # noqa: E402
from depthwizard.calibrate import fit_calibration  # noqa: E402
from depthwizard.config import CALIBRATION_PATH  # noqa: E402
from depthwizard.depth import DepthEstimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=20)
    ap.add_argument("--cities", default="DC,PHL,NYC")
    ap.add_argument("--out", default=CALIBRATION_PATH)
    ap.add_argument("--cache", default=str(ROOT / "results" / "calib_cache_finetuned"))
    args = ap.parse_args()
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    import os

    est = DepthEstimator()
    print(f"estimator ready ({est.device})")

    heldout_ids = {tid for _, tid in _gamus.heldout_entries()}
    tiles = [(c.strip().upper(), t)
             for c in args.cities.split(",")
             for t in _gamus.train_tiles(c.strip().upper(), exclude=heldout_ids, limit=args.per_city)]
    print(f"fitting on {len(tiles)} train tiles", flush=True)

    dw, nw, city_counts = [], [], {}
    for city, tid in tiles:
        rgb = _gamus.tile_rgb(city, "train", tid)
        ndsm = _gamus.tile_ndsm(city, "train", tid)
        cf = Path(args.cache) / f"{city}_{tid}.npy"
        if cf.exists():
            depth = np.load(cf)
        else:
            depth = est.predict(rgb)
            np.save(cf, depth)
        h = min(depth.shape[0], ndsm.shape[0])
        w = min(depth.shape[1], ndsm.shape[1])
        dw.append(depth[:h, :w]); nw.append(ndsm[:h, :w])
        city_counts[city] = city_counts.get(city, 0) + 1

    calib = fit_calibration(dw, nw)
    calib.city_counts = city_counts
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    calib.to_json(args.out)
    print(f"\nFine-tuned calibration: slope={calib.slope:.4f} intercept={calib.intercept:.3f}")
    print(f"pixels={calib.n_pixels} tiles={calib.n_tiles} cities={city_counts}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

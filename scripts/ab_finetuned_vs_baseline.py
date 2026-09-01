"""Honest A/B: frozen baseline vs fine-tuned backbone, SAME calibration protocol.

The shipped baseline = frozen DA-V2 depth + RANSAC global linear calibration fit
on GAMUS train. This script tests the one scientific question from the fine-tune
experiment that might still help: does the *fine-tuned* backbone supply a stronger
structural prior that a recalibrated linear map can anchor better than the frozen
one?

Protocol (identical to baseline, only the depth signal changes):
  - fit: fine-tuned depth on ``--fit-per-city`` fresh GAMUS train tiles ->
         RANSAC slope/intercept via depthwizard.calibrate.fit_calibration
  - eval: fine-tuned depth on the 9 held-out tiles -> predicted nDSM = a*depth+b
         vs GT AGL nDSM  (same comparator as validate_held_out.py)

Usage (DW_FINETUNED must point at the fine-tuned weights):
    DW_FINETUNED=models/ft-report prime-run python scripts/ab_finetuned_vs_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _gamus  # noqa: E402
from depthwizard.calibrate import fit_calibration  # noqa: E402
from depthwizard.depth import DepthEstimator  # noqa: E402
from depthwizard.validate import compare  # noqa: E402

VALIDATE_DIR = ROOT / "results" / "heldout"   # baseline depth cache (frozen) not needed here
DATA_DIR = ROOT / "data" / "ft-heldout"        # held-out rgb + ndsm already staged there
OUT_DIR = ROOT / "results" / "ab_finetuned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-per-city", type=int, default=8)
    ap.add_argument("--cities", default="DC,PHL,NYC")
    ap.add_argument("--cache", default=str(OUT_DIR / "depth_cache"))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    est = DepthEstimator()
    print(f"estimator ready ({est.device})")

    # ---- fit calibration on fine-tuned depth over fresh train tiles ----
    heldout_ids = {tid for _, tid in _gamus.heldout_entries()}
    fit_tiles = [(c.strip().upper(), t) for c in args.cities.split(",")
                 for t in _gamus.train_tiles(c.strip().upper(), exclude=heldout_ids, limit=args.fit_per_city)]
    print(f"fit on {len(fit_tiles)} train tiles", flush=True)
    dw, nw = [], []
    for city, tid in fit_tiles:
        rgb = _gamus.tile_rgb(city, "train", tid)
        ndsm = _gamus.tile_ndsm(city, "train", tid)
        cf = cache_dir / f"fit_{city}_{tid}.npy"
        if cf.exists():
            depth = np.load(cf)
        else:
            depth = est.predict(rgb)
            np.save(cf, depth)
        h = min(depth.shape[0], ndsm.shape[0])
        w = min(depth.shape[1], ndsm.shape[1])
        dw.append(depth[:h, :w]); nw.append(ndsm[:h, :w])
    calib = fit_calibration(dw, nw)
    print(f"fine-tuned calibration: slope={calib.slope:.4f} intercept={calib.intercept:.3f} n_pix={calib.n_pixels}", flush=True)

    # ---- eval on held-out ----
    rows = []
    for city, tid in _gamus.heldout_entries():
        png = DATA_DIR / f"{tid}.png"
        ref_tif = DATA_DIR / f"{tid}_ndsm.tif"
        with rasterio.open(ref_tif) as src:
            ref = src.read(1).astype(np.float64)
        depth = est.predict(np.asarray(__import__("PIL").Image.open(png).convert("RGB")))
        h = min(depth.shape[0], ref.shape[0]); w = min(depth.shape[1], ref.shape[1])
        depth, ref = depth[:h, :w], ref[:h, :w]
        pred = calib.slope * depth + calib.intercept
        m = compare(pred, ref)
        m["tile_id"] = tid; m["city"] = city
        rows.append(m)
        print(f"  [{city}] {tid}: RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} Pearson={m['pearson']:.3f}", flush=True)

    summary = {
        "fit_per_city": args.fit_per_city, "n_fit_tiles": len(fit_tiles),
        "calib": {"slope": calib.slope, "intercept": calib.intercept, "n_pixels": calib.n_pixels},
        "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
        "mae_mean": float(np.mean([r["mae"] for r in rows])),
        "pearson_mean": float(np.mean([r["pearson"] for r in rows])),
        "per_tile": rows,
        "baseline_ref": {"rmse_mean": 5.63, "mae_mean": 4.51, "pearson_mean": 0.255},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== A/B: fine-tuned backbone + recalibrated =====")
    print(f"  RMSE={summary['rmse_mean']:.2f} MAE={summary['mae_mean']:.2f} Pearson={summary['pearson_mean']:.3f}")
    print(f"  (baseline frozen = RMSE 5.63 MAE 4.51 Pearson 0.255)")
    print(f"saved -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

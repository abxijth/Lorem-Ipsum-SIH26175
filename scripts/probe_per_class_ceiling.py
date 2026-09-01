"""Oracle per-class ceiling probe (A1 gate).

Fits a GLOBAL and a PER-CLASS calibration on a handful of freshly sampled GAMUS
train tiles, then applies BOTH to the already-computed held-out depths
(results/heldout/**/*_depth.npy) using the GROUND-TRUTH held-out class maps.

Per-class with oracle labels is the best case a segmentation head could give us.
If it does not beat the global line by >= --gate RMSE improvement, training a
segmenter for conditioning is not worth it and the gate will tell us to stop.

Usage:
    prime-run python scripts/probe_per_class_ceiling.py [--per-city 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _gamus  # noqa: E402
from depthwizard.calibrate import apply_calibration, fit_calibration, fit_calibration_classwise  # noqa: E402
from depthwizard.depth import DepthEstimator  # noqa: E402
from depthwizard.validate import compare  # noqa: E402

VALIDATE_DIR = ROOT / "results" / "heldout"
DATA_DIR = ROOT / "data" / "heldout"
OUT_DIR = ROOT / "results" / "probe"


def per_class_rmse(pred: np.ndarray, ref: np.ndarray, cls: np.ndarray,
                   labels=(3, 1, 6, 2, 5)) -> dict:
    """RMSE per semantic class (valid, non-nodata pixels)."""
    out = {}
    for label in labels:
        m = (cls == label) & np.isfinite(pred) & np.isfinite(ref) & (ref >= 0)
        if m.sum() == 0:
            out[label] = {"n": 0, "rmse": None}
        else:
            err = ref[m] - pred[m]
            out[label] = {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean(err ** 2)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=2, help="train tiles sampled per city")
    ap.add_argument("--cities", default="DC,PHL,NYC")
    ap.add_argument("--gate", type=float, default=0.10, help="min relative RMSE gain to proceed")
    ap.add_argument("--class-set", default="",
                    help="comma-separated labels to fit per-class (default: all non-0 labels present)")
    ap.add_argument("--save-v2", default="", help="path to write the fitted per-class calibration JSON")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "depth_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    est = DepthEstimator()
    print(f"depth estimator ready ({est.device})")

    heldout_ids = {tid for _, tid in _gamus.heldout_entries()}
    train_tiles = [(c, t) for c in args.cities.split(",")
                   for t in _gamus.train_tiles(c.strip().upper(), exclude=heldout_ids, limit=args.per_city)]
    print(f"training on {len(train_tiles)} fresh train tiles: {train_tiles}")

    dw, nw, cw = [], [], []
    for city, tid in train_tiles:
        t0 = time.time()
        rgb = _gamus.tile_rgb(city, "train", tid)
        ndsm = _gamus.tile_ndsm(city, "train", tid)
        cls = _gamus.tile_cls(city, "train", tid)
        cache = cache_dir / f"{city}_{tid}_depth.npy"
        if cache.exists():
            depth = np.load(cache)
        else:
            depth = est.predict(rgb)
            np.save(cache, depth)
        h = min(depth.shape[0], ndsm.shape[0], rgb.shape[0])
        w = min(depth.shape[1], ndsm.shape[1], rgb.shape[1])
        depth = depth[:h, :w]
        ndsm = ndsm[:h, :w]
        cls = cls[:h, :w]
        dw.append(depth)
        nw.append(ndsm)
        cw.append(cls)
        print(f"  {city} {tid} depth computed ({time.time()-t0:.0f}s)")

    calib_global = fit_calibration(dw, nw)
    classes = [int(x) for x in args.class_set.split(",") if x.strip()] if args.class_set else None
    calib_per = fit_calibration_classwise(dw, nw, cw, classes=classes)
    print(f"\nglobal: slope={calib_global.slope:.3f} intercept={calib_global.intercept:.2f}")
    print("per-class:")
    for k in sorted(calib_per.class_slopes, key=int):
        print(f"  {int(k)} {calib_per.class_slopes[k]:+.3f}*d +{calib_per.class_intercepts[k]:+.2f}"
              f" (n={calib_per.class_counts[k]})")

    # ---- apply both to held-out depths with oracle class maps ----
    rows, per_agg = [], {"global": {}, "oracle": {}}
    for city, tid in _gamus.heldout_entries():
        depth = np.load(VALIDATE_DIR / city / f"{tid}_depth.npy")
        with rasterio.open(DATA_DIR / f"{tid}_ndsm.tif") as src:
            ref = src.read(1).astype(np.float64)
        h = min(depth.shape[0], ref.shape[0])
        w = min(depth.shape[1], ref.shape[1])
        depth, ref = depth[:h, :w], ref[:h, :w]
        cls = _gamus.tile_cls(city, _gamus.SPLIT[city], tid)[:h, :w]

        pred_g = apply_calibration(depth, calib_global)
        pred_o = apply_calibration(depth, calib_per, cls)
        m_g = compare(pred_g, ref)
        m_o = compare(pred_o, ref)
        rows.append({
            "city": city, "tile": tid,
            "global_rmse": m_g["rmse"], "global_pearson": m_g["pearson"],
            "oracle_rmse": m_o["rmse"], "oracle_pearson": m_o["pearson"],
        })
        for key, pred in (("global", pred_g), ("oracle", pred_o)):
            for label, v in per_class_rmse(pred, ref, cls).items():
                per_agg[key].setdefault(label, []).append(v["rmse"] if v["rmse"] is not None else None)
        print(f"  [{city}] {tid}: global RMSE={m_g['rmse']:.2f} | oracle per-class RMSE={m_o['rmse']:.2f}")

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals))

    summary = {
        "n_train_tiles": len(train_tiles),
        "global_slope": calib_global.slope, "global_intercept": calib_global.intercept,
        "per_class_slopes": calib_per.class_slopes,
        "global": {"rmse": avg("global_rmse"), "pearson": avg("global_pearson")},
        "oracle": {"rmse": avg("oracle_rmse"), "pearson": avg("oracle_pearson")},
        "per_class_rmse": per_agg,
        "per_tile": rows,
    }
    gain = 1.0 - summary["oracle"]["rmse"] / summary["global"]["rmse"]
    summary["relative_rmse_gain"] = gain
    summary["proceed"] = gain >= args.gate
    (OUT_DIR / "ceiling.json").write_text(json.dumps(summary, indent=2))

    if args.save_v2:
        out_v2 = Path(args.save_v2)
        out_v2.parent.mkdir(parents=True, exist_ok=True)
        calib_per.city_counts = {c: args.per_city for c in args.cities.split(",")}
        calib_per.to_json(out_v2)
        print(f"per-class calibration saved -> {out_v2}")

    print("\n===== ORACLE CEILING (fit=train, eval=held-out, oracle class maps) =====")
    print(f"  global   RMSE={summary['global']['rmse']:.2f} m  Pearson={summary['global']['pearson']:.3f}")
    print(f"  oracle   RMSE={summary['oracle']['rmse']:.2f} m  Pearson={summary['oracle']['pearson']:.3f}")
    print(f"  relative RMSE gain = {gain*100:.1f}%  (gate {args.gate*100:.0f}%) -> {'PROCEED' if summary['proceed'] else 'STOP'}")
    print(f"saved -> {OUT_DIR / 'ceiling.json'}")


if __name__ == "__main__":
    main()
"""Evaluate a piecewise-linear depth->height calibration on held-out tiles.

Fits both a linear (global) and a piecewise calibration on the SAME 30-tile
train pool used by the oracle probe (cached depths, no GPU needed), then applies
them to the held-out depths. Compares against the shipped global calibration
(models/calibration.json).

Usage:
    python scripts/eval_piecewise.py [--per-city 10] [--segments 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _gamus  # noqa: E402
from depthwizard.calibrate import (  # noqa: E402
    apply_calibration_any, fit_calibration, fit_piecewise, fit_piecewise_segments,
    load_calibration_any,
)
from depthwizard.validate import compare  # noqa: E402

PROBE_DEPTHS = ROOT / "results" / "probe" / "depth_cache"
VALIDATE_DIR = ROOT / "results" / "heldout"
DATA_DIR = ROOT / "data" / "heldout"
OUT_DIR = ROOT / "results" / "piecewise"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=10)
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--mode", choices=["median", "segments"], default="median",
                    help="anchor method: median height per bin, or each bin's RANSAC line")
    ap.add_argument("--out", default=str(ROOT / "models" / "calibration_piecewise.json"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    heldout_ids = {tid for _, tid in _gamus.heldout_entries()}
    train_tiles = [(c, t) for c in _gamus.CITIES for t in
                   _gamus.train_tiles(c, exclude=heldout_ids, limit=args.per_city)]
    print(f"train pool: {len(train_tiles)} tiles ({args.per_city}/city)")

    dw, nw = [], []
    for city, tid in train_tiles:
        depth = np.load(PROBE_DEPTHS / f"{city}_{tid}_depth.npy")
        ndsm = _gamus.tile_ndsm(city, "train", tid)
        h = min(depth.shape[0], ndsm.shape[0])
        w = min(depth.shape[1], ndsm.shape[1])
        dw.append(depth[:h, :w])
        nw.append(ndsm[:h, :w])

    lin = fit_calibration(dw, nw)
    if args.mode == "segments":
        pw = fit_piecewise_segments(dw, nw, n_segments=args.segments)
    else:
        pw = fit_piecewise(dw, nw, n_segments=args.segments)
    print(f"\nlinear   : slope={lin.slope:.3f} intercept={lin.intercept:.2f}")
    print("piecewise anchors (depth -> height):")
    for a, v in zip(pw.anchors, pw.values):
        print(f"    d={a:6.3f} -> h={v:5.2f} m")

    # reference numbers from the shipped global model
    shipped = load_calibration_any(ROOT / "models" / "calibration.json")

    by = {"shipped": {}, "linear30": {}, "piecewise": {}}
    rows = []
    for city, tid in _gamus.heldout_entries():
        depth = np.load(VALIDATE_DIR / city / f"{tid}_depth.npy")
        with rasterio.open(DATA_DIR / f"{tid}_ndsm.tif") as src:
            ref = src.read(1).astype(np.float64)
        h = min(depth.shape[0], ref.shape[0])
        w = min(depth.shape[1], ref.shape[1])
        depth, ref = depth[:h, :w], ref[:h, :w]
        res = {}
        for key, calib in (("shipped", shipped), ("linear30", lin), ("piecewise", pw)):
            m = compare(apply_calibration_any(depth, calib), ref)
            res[key] = {"rmse": m["rmse"], "mae": m["mae"], "pearson": m["pearson"]}
            by[key].setdefault(city, []).append(m["rmse"])
        rows.append({"tile": f"{city}:{tid}", **{f"{k}_rmse": v["rmse"] for k, v in res.items()}})
        print(f"  {city:3s} {tid}: shipped={res['shipped']['rmse']:.2f} "
              f"linear30={res['linear30']['rmse']:.2f} piecewise={res['piecewise']['rmse']:.2f} m")

    def avg(key):
        return float(np.mean([r[f"{key}_rmse"] for r in rows]))

    def city_avg(key, city):
        return float(np.mean(by[key][city]))

    summary = {
        "segments": args.segments,
        "n_train_tiles": len(train_tiles),
        "anchors": pw.anchors, "values": pw.values,
        "overall": {"shipped": avg("shipped"), "linear30": avg("linear30"), "piecewise": avg("piecewise")},
        "per_city": {
            c: {
                "shipped": city_avg("shipped", c),
                "linear30": city_avg("linear30", c),
                "piecewise": city_avg("piecewise", c),
            } for c in sorted(by["shipped"])
        },
        "per_tile": rows,
    }
    overall = summary["overall"]
    gain_vs_shipped = 1.0 - overall["piecewise"] / overall["shipped"]
    regressions = {
        c: city_avg("piecewise", c) - city_avg("shipped", c)
        for c in summary["per_city"] if city_avg("piecewise", c) > city_avg("shipped", c)
    }
    summary["gain_vs_shipped"] = gain_vs_shipped
    summary["regressed_cities"] = regressions
    summary["promote"] = gain_vs_shipped >= 0.05 and not regressions
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pw.city_counts = {c: args.per_city for c in _gamus.CITIES}
    pw.to_json(args.out)
    (OUT_DIR / f"summary_{args.mode}.json").write_text(json.dumps(summary, indent=2))

    print("\n===== PIECEWISE vs GLOBAL (held-out) =====")
    print(f"  shipped global RMSE : {overall['shipped']:.2f} m")
    print(f"  30-tile linear RMSE : {overall['linear30']:.2f} m")
    print(f"  piecewise({args.mode}) RMSE  : {overall['piecewise']:.2f} m  (gain {gain_vs_shipped*100:.1f}%)")
    for c, d in summary["per_city"].items():
        print(f"    {c}: shipped {d['shipped']:.2f} -> piecewise {d['piecewise']:.2f} m")
    print(f"  regressed cities: {regressions or 'none'}")
    print(f"  promote to default: {summary['promote']}")
    print(f"saved -> {args.out}")
    print(f"summary -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
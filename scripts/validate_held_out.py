"""Honest held-out validation of the calibration head.

Fits are trained on GAMUS train (`scripts/fit_calibration.py`); this script
runs the full pipeline on *held-out* tiles (val split for DC/PHL, test for NYC)
and reports RMSE / MAE / Pearson of predicted structural height vs. GT nDSM.

Georeference is intentionally omitted so the pipeline takes the relative-only
path: predicted DSM == structural height, directly comparable to AGL nDSM.

Usage:
    prime-run python scripts/validate_held_out.py [--tiles "DC:DC_11_16,PHL:PHL_6150,..."]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from depthwizard import pipeline  # noqa: E402

REPO = "earthflow/GAMUS"
IMG_SUFFIX = {"DC": "_RGB.h5", "PHL": "_RGB.h5", "NYC": "_IMG.h5"}
# NYC has no val split -> hold out from test.
SPLIT = {"DC": "val", "PHL": "val", "NYC": "test"}

DEFAULT = [
    "DC:DC_11_16", "DC:DC_12_17", "DC:DC_13_14",
    "PHL:PHL_6150", "PHL:PHL_6151", "PHL:PHL_6152",
    "NYC:NYC_00918", "NYC:NYC_00921", "NYC:NYC_00735",
]


def grab(split: str, tid: str, image: bool, city: str):
    """Download one GAMUS h5. image=True for RGB (needs city-specific suffix)."""
    if image:
        fn = f"{tid}{IMG_SUFFIX[city]}"
        folder = "images"
    else:
        fn = f"{tid}_AGL.h5"
        folder = "heights"
    p = hf_hub_download(REPO, f"{folder}/{split}/{fn}", repo_type="dataset")
    with h5py.File(p, "r") as f:
        return f["image"][...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=",".join(DEFAULT), help="comma-sep CITY:TILE entries")
    ap.add_argument("--calibration", default="models/calibration.json")
    ap.add_argument("--out", default="results/heldout")
    ap.add_argument("--tmp", default="data/heldout")
    args = ap.parse_args()

    tmp = Path(args.tmp)
    out = Path(args.out)
    tmp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    entries = [(t.split(":")[0].strip(), t.split(":")[1].strip()) for t in args.tiles.split(",")]
    metrics = []

    for city, tid in entries:
        split = SPLIT[city]
        png = tmp / f"{tid}.png"
        ref_tif = tmp / f"{tid}_ndsm.tif"

        rgb = grab(split, tid, image=True, city=city)
        ndsm = grab(split, tid, image=False, city=city).astype(np.float32)
        Image.fromarray(rgb).save(png)
        with rasterio.open(ref_tif, "w", driver="GTiff", height=ndsm.shape[0], width=ndsm.shape[1],
                            count=1, dtype="float32") as dst:
            dst.write(ndsm, 1)

        res = pipeline.run(png, reference_path=ref_tif, calibration_path=args.calibration, out_dir=out / city)
        m = res.metrics
        m["tile_id"] = tid
        m["city"] = city
        metrics.append(m)
        print(f"  [{city}] {tid}: RMSE={m['rmse']:.2f}m MAE={m['mae']:.2f}m "
              f"Pearson={m['pearson']:.3f} n={m['n']}")

    rmse = [m["rmse"] for m in metrics]
    mae = [m["mae"] for m in metrics]
    prs = [m["pearson"] for m in metrics]
    import json
    summary = {
        "n_tiles": len(metrics),
        "rmse_mean": float(np.mean(rmse)),
        "mae_mean": float(np.mean(mae)),
        "pearson_mean": float(np.mean(prs)),
        "per_tile": metrics,
    }
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== HELD-OUT SUMMARY (fit=train, eval=held-out) =====")
    print(f"  RMSE   mean={np.mean(rmse):.2f} m")
    print(f"  MAE    mean={np.mean(mae):.2f} m")
    print(f"  Pearson mean={np.mean(prs):.3f}")
    print(f"saved -> {out / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
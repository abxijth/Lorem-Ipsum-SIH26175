"""CLI entrypoint for the DepthWizard backend.

Usage:
    prime-run python -m depthwizard.cli INPUT [--bbox "W S E N"] [--reference GT.tif]
                                             [--calibration path.json] [--gcp "x,y,h;..."]

Examples:
    python -m depthwizard.cli data/tile.png --bbox "38.895 -77.03 38.915 -77.01"
    python -m depthwizard.cli input.tif --reference ref.tif
    python -m depthwizard.cli clip.png --gcp "120,50,18;300,400,3"   # click-based refit
"""

from __future__ import annotations

import argparse
import json
import sys

from . import pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m depthwizard.cli", description=__doc__)
    ap.add_argument("input", help="input image or GeoTIFF")
    ap.add_argument("--bbox", help="WGS84 bounds as 'west south east north'")
    ap.add_argument("--reference", help="reference elevation GeoTIFF for validation")
    ap.add_argument("--calibration", help="calibration JSON (default models/calibration.json)")
    ap.add_argument("--gcp", help="ground-control points as 'x,y,height;x,y,height' "
                                  "(pixel coords + known height; refits calibration)")
    ap.add_argument("--out", help="output directory (default results/output)")
    args = ap.parse_args(argv)

    try:
        result = pipeline.run(
            input_path=args.input,
            bbox=args.bbox,
            reference_path=args.reference,
            calibration_path=args.calibration,
            out_dir=args.out,
            gcp=args.gcp,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(json.dumps(vars(result), indent=2, default=str))
    if result.metrics:
        m = result.metrics
        print(f"\nRMSE: {m['rmse']:.2f} m | MAE: {m['mae']:.2f} m | "
              f"Pearson r: {m['pearson']:.3f} | n: {m['n']}")
    for w in result.warnings:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
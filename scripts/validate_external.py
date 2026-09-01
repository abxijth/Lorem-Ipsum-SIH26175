"""Run the DepthWizard pipeline on external landscape cases and report metrics.

Reads data/external/<case>/ (input.png + reference.tif + meta.json), runs the
shipped pipeline on each, and aggregates RMSE/MAE/Pearson per case. Where a
Copernicus GLO-30 surface.tif also exists (forest case), it quantifies the
canopy gap (surface - bare-earth terrain) and the model's surface agreement.

Outputs:
  results/external/<case>/<case>_report.json   (pipeline result)
  results/external/<case>/<case>_preview.png   (RGB | predicted | reference | diff)
  results/external/summary.json                (aggregate table)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))  # noqa: E402

DATA = ROOT / "data" / "external"
OUT = ROOT / "results" / "external"
CALIB = ROOT / "models" / "calibration.json"

from depthwizard import pipeline, validate


def _load_flat(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _align_to(ref_path: Path, target: np.ndarray, src_path: Path) -> np.ndarray:
    """Resample src raster onto target's grid (same CRS assumed)."""
    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        out = np.empty_like(target, dtype=np.float32)
        reproject(
            source=src.read(1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=-9999,
            resampling=Resampling.bilinear,
        )
    return out


def run_case(case_dir: Path) -> dict:
    name = case_dir.name
    meta = json.loads((case_dir / "meta.json").read_text())
    input_img = case_dir / "input.png"
    ref = case_dir / "reference.tif"
    bbox = ",".join(str(x) for x in meta["bbox"])
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] running pipeline ...", flush=True)
    res = pipeline.run(
        input_img, bbox=bbox, reference_path=ref,
        calibration_path=CALIB, out_dir=out_dir,
    )
    record = {
        "name": name,
        "landscape": meta["landscape"],
        "bbox": meta["bbox"],
        "metrics": res.metrics,
        "calibration": res.calibration_path,
        "warnings": res.warnings,
    }

    # Canopy gap + surface agreement where a GLO-30 surface is available.
    surface = case_dir / "surface.tif"
    if surface.exists():
        dsm = _load_flat(out_dir / f"{input_img.stem}_dsm.tif")
        terrain = _load_flat(ref)
        surf = _align_to(ref, terrain, surface)
        canopy = surf - terrain  # surface above bare earth
        record["_terrain"] = {
            "range_m": [float(terrain.min()), float(terrain.max())],
            "rmse_m": float(np.sqrt(np.mean((surf - terrain) ** 2))),
            "canopy_mean_m": float(np.mean(canopy)),
            "canopy_p95_m": float(np.percentile(canopy, 95)),
        }
        record["_surface_metrics"] = validate.compare(dsm, surf)

    # 4-panel preview (RGB | predicted DSM | reference terrain | diff)
    _preview(name, case_dir, input_img, out_dir, ref)

    return record


def _preview(name: str, case_dir: Path, input_img: Path, out_dir: Path, ref: Path) -> None:
    rgb = np.asarray(__import__("PIL").Image.open(input_img).convert("RGB"))
    dsm_path = out_dir / f"{input_img.stem}_dsm.tif"
    dsm = _load_flat(dsm_path)
    terrain = _load_flat(ref)
    err = np.abs(dsm - terrain)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a in ax:
        a.set_axis_off()
    ax[0].imshow(rgb); ax[0].set_title("Input RGB (NAIP-derived)")
    p = ax[1].imshow(dsm, cmap="terrain"); ax[1].set_title("Predicted DSM (m)")
    fig.colorbar(p, ax=ax[1], fraction=0.046)
    q = ax[2].imshow(terrain, cmap="terrain"); ax[2].set_title("Reference DEM (bare earth, m)")
    fig.colorbar(q, ax=ax[2], fraction=0.046)
    r = ax[3].imshow(err, cmap="hot", vmax=min(float(np.percentile(err, 98)), 60))
    ax[3].set_title("|error| heatmap (m)")
    fig.colorbar(r, ax=ax[3], fraction=0.046)
    fig.suptitle(f"{name} — {json.loads((case_dir/'meta.json').read_text())['landscape']}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_preview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cases = sorted(p for p in DATA.iterdir() if (p / "input.png").exists())
    if not cases:
        raise SystemExit("no cases under data/external/ — run scripts/fetch_external_data.py first")
    records = [run_case(c) for c in cases]

    summary = {r["name"]: r for r in records}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== LANDSCAPE VALIDATION SUMMARY ===")
    print(f"{'case':<16}{'landscape':<10}{'n':<12}{'RMSE(m)':<9}{'MAE(m)':<8}{'Pearson':<9}")
    for r in records:
        m = r["metrics"]
        print(f"{r['name']:<16}{r['landscape']:<10}{m['n']:<12}{m['rmse']:<9.2f}{m['mae']:<8.2f}{m['pearson']:<9.3f}")


if __name__ == "__main__":
    main()
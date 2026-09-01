"""End-to-end pipeline orchestrator.

    upload -> georef -> depth -> calibration -> [SRTM baseline] -> DSM -> GeoTIFF
    optional reference -> RMSE/MAE/Pearson + error heatmap

Used by the CLI now, and by Person B's FastAPI /process later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import calibrate
from .calibrate import Calibration, load_calibration_any
from .config import RESULTS_DIR
from .depth import DepthEstimator
from .dsm import assemble, write_geotiff
from . import georef, srtm, validate


@dataclass
class PipelineResult:
    input: str
    georeferenced: bool
    crs: str | None
    bbox_wgs84: list | None
    depth_path: str | None = None
    dsm_path: str | None = None
    heatmap_path: str | None = None
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    calibration_path: str | None = None


def _save_npy(arr: np.ndarray, path: Path | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return str(path)


def run(
    input_path: Path | str,
    bbox: str | None = None,
    reference_path: Path | str | None = None,
    calibration_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    model_id: str | None = None,
    gcp: str | None = None,
) -> PipelineResult:
    """Run the full pipeline on one input image. Returns paths + metrics."""
    input_path = Path(input_path)
    out = Path(out_dir) if out_dir else (RESULTS_DIR / "output")
    out.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    warnings: list[str] = []

    # 1. Input
    gr = georef.load(input_path, bbox)

    # 2. Relative depth
    est = DepthEstimator(model_id=model_id) if model_id else DepthEstimator()
    depth = est.predict(gr.rgb)
    depth_path = _save_npy(depth, out / f"{stem}_depth.npy")

    # 3. Calibration (loaded, default-normalized, or refit from user GCPs)
    try:
        calib = load_calibration_any(calibration_path)
        calib_src = str(calibration_path)
    except FileNotFoundError:
        calib = _default_normalization(depth)
        calib_src = None
        warnings.append("no calibration found -> normalized relative heights (not metres)")

    if gcp:
        points, heights = calibrate.parse_gcp(gcp)
        calib = calibrate.gcp_refit(depth, points, heights, base_calib=calib)
        calib_src = f"gcp ({len(points)} point{'s' if len(points) != 1 else ''})"
        if len(points) < 2:
            warnings.append("1 GCP -> intercept-only shift (slope kept from base calibration)")

    # 4a. SRTM terrain baseline (georeferenced inputs only)
    terrain = None
    if gr.bbox_wgs84 and gr.crs:
        try:
            terrain = srtm.terrain_baseline(gr.bbox_wgs84, gr.crs, gr.transform, depth.shape)
        except Exception as e:
            warnings.append(f"SRTM failed ({e}); structural-height only")
            terrain = None

    # 4b. Assemble absolute DSM + export
    dsm = assemble(depth, calib, terrain)
    dsm_path = write_geotiff(dsm, gr, out / f"{stem}_dsm.tif")

    # 5. Validation (if reference provided)
    metrics = {}
    heatmap_path = None
    if reference_path:
        ref, _ = validate.load_raster_flat(reference_path)
        if ref.shape != dsm.shape:
            warnings.append(
                f"reference shape {ref.shape} != DSM {dsm.shape}; resizing reference "
                "(differential resample, best effort)"
            )
            from PIL import Image
            ref = np.asarray(Image.fromarray(ref).resize((dsm.shape[1], dsm.shape[0]), Image.BILINEAR))
        metrics = validate.compare(dsm, ref)
        heatmap_path = validate.error_heatmap(dsm, ref, gr, out / f"{stem}_error.tif")

    result = PipelineResult(
        input=str(input_path),
        georeferenced=bool(gr.crs),
        crs=gr.crs,
        bbox_wgs84=list(gr.bbox_wgs84) if gr.bbox_wgs84 else None,
        depth_path=depth_path,
        dsm_path=dsm_path,
        heatmap_path=heatmap_path,
        metrics=metrics,
        warnings=warnings,
        calibration_path=calib_src,
    )

    (out / f"{stem}_report.json").write_text(json.dumps(asdict(result), indent=2))
    return result


def _default_normalization(depth: np.ndarray) -> Calibration:
    """Relative-only fallback: squashes depth into a plausible 0-40 m visual range."""
    d = depth.astype(np.float64)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-9:
        slope, intercept = 0.0, 0.0
    else:
        slope = 40.0 / (hi - lo)
        intercept = -slope * lo
    return Calibration(slope=slope, intercept=intercept, n_pixels=0, n_tiles=-1, city_counts={})
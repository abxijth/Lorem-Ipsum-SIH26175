"""Validation metrics against reference elevation.

RMSE (m), MAE (m), Pearson r and a colorized error-heatmap GeoTIFF
(|reference - predicted|) written with the same georeference as the DSM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import pearsonr

from .dsm import NODATA, write_geotiff
from .georef import Georef


def compare(pred: np.ndarray, reference: np.ndarray) -> dict:
    """Metrics on valid (finite, non-nodata) pixel pairs. Same grid assumed."""
    p = pred.astype(np.float64)
    r = reference.astype(np.float64)
    valid = np.isfinite(p) & np.isfinite(r) & (p != NODATA) & (r != NODATA)
    p, r = p[valid], r[valid]
    if p.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "pearson": np.nan, "bias": np.nan}
    err = r - p
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(p - r))  # positive = overestimate
    if np.std(p) < 1e-12 or np.std(r) < 1e-12:
        rr = np.nan
    else:
        rr = float(pearsonr(p, r)[0])
    return {"n": int(p.size), "rmse": rmse, "mae": mae, "pearson": rr, "bias": bias}


def error_heatmap(pred: np.ndarray, reference: np.ndarray, georef: Georef, out_path: Path | str) -> str:
    """Write a colorized |reference - pred| error map GeoTIFF (Float32, metres)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err = np.abs(reference.astype(np.float32) - pred.astype(np.float32))
    err[~np.isfinite(err)] = NODATA
    err[pred == NODATA] = NODATA
    return write_geotiff(err, georef, out_path)


def load_raster_flat(path: Path | str) -> tuple[np.ndarray, dict]:
    """Read a single-band raster as 2D float32."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile
"""DSM assembly + GeoTIFF export.

Absolute DSM = SRTM terrain baseline + calibrated structural height.

The structural-height head reads size/structure from the depth map (buildings,
trees); SRTM supplies the ground elevation beneath them. For the relative-only
path (no georeference) there is no SRTM, so the DSM is just the normalized
structural height (plausible visual scale).

Export convention: Float32 GeoTIFF, WGS84 (or the image's CRS), nodata=-9999.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from .calibrate import Calibration, apply_calibration_any
from .georef import Georef

NODATA = -9999.0


def structural_height(depth: np.ndarray, calib: Calibration, class_map: np.ndarray | None = None) -> np.ndarray:
    return apply_calibration_any(depth, calib, class_map)


def assemble(depth: np.ndarray, calib: Calibration, terrain: np.ndarray | None,
             class_map: np.ndarray | None = None) -> np.ndarray:
    """Absolute DSM (m). Without terrain, falls back to structural height."""
    struct_h = structural_height(depth, calib, class_map)
    if terrain is None:
        return struct_h
    return terrain + struct_h


def _default_affine(h: int, w: int) -> Affine:
    """For nongeoreferenced output: a unit-size pixel grid."""
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, h)


def write_geotiff(dsm: np.ndarray, georef: Georef, out_path: Path | str,
                  dtype: str | None = None, nodata: float | None = None) -> str:
    """Write the DSM as a Float32 GeoTIFF using the input's CRS+transform.

    `dtype`/`nodata` override the defaults (e.g. uint8 + 0 for class maps).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    crs = georef.crs
    transform = georef.transform if georef.transform is not None else _default_affine(*dsm.shape)

    dtype = dtype or rasterio.float32
    nodata = NODATA if nodata is None else nodata
    profile = {
        "driver": "GTiff",
        "height": dsm.shape[0],
        "width": dsm.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    arr = np.asarray(dsm)
    if arr.dtype != np.dtype(dtype):
        arr = arr.astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.floating):
        arr = np.where(np.isfinite(arr), arr, nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    return str(out_path)
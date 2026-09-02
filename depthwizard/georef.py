"""Input handling: turn an uploaded file into (rgb array, georeference).

Three cases:
  * GeoTIFF          -> rasterio reads CRS + transform directly (georeferenced);
                        WGS84 bounds derived for SRTM fetching
  * PNG/JPG + bbox   -> bbox given as WGS84 (west south east north); we build a
                        local UTM affine so metric math (SRTM resampling, DSM)
                        is done in metres
  * PNG/JPG only     -> relative-only path (no georeference, no SRTM)

GeoTIFF output is always written in WGS84 so it opens cleanly in any GIS tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from PIL import Image


@dataclass
class Georef:
    rgb: np.ndarray            # (H, W, 3) uint8
    crs: str | None = None     # e.g. "EPSG:32618"; None when nongeoreferenced
    transform: object | None = None   # rasterio Affine describing rgb coordinate space
    bbox_wgs84: tuple | None = None   # (west, south, east, north); None when unknown
    source: str = "png"        # original file suffix
    meta: dict = field(default_factory=dict)
    agl: bool = False          # AGL (above-ground-level) heights: nDSM is structural
                               # only. Georeferenced (so GSD/slope work) but the DSM
                               # must NOT have the SRTM ground baseline added.


def _read_image(path: Path) -> np.ndarray:
    """Read any file as a uint8 RGB (H, W, 3) array (PNG/JPG/GeoTIFF)."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff", ".gtiff"}:
        try:
            with rasterio.open(path) as src:
                if src.count >= 3:
                    return np.dstack([src.read(i) for i in (1, 2, 3)]).astype(np.uint8)
        except Exception:
            pass
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).astype(np.uint8)


def load_georeferenced(path: Path) -> Georef:
    """Load a georeferenced file (GeoTIFF)."""
    with rasterio.open(path) as src:
        crs = str(src.crs) if src.crs else None
        transform = src.transform
        meta = src.meta.copy()
        if crs:
            bbox = tuple(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
        else:
            bbox = None
        tags = src.tags()
    rgb = _read_image(path)
    # Optional AGL (above-ground-level) marker: tile carries structural heights
    # (e.g. GAMUS nDSM), so skip the SRTM ground baseline downstream.
    agl = str(tags.get("GAMUS", "")).strip().upper() == "AGL"
    return Georef(rgb=rgb, crs=crs, transform=transform, bbox_wgs84=bbox,
                  source=path.suffix.lower(), meta=meta, agl=agl)


def load_bbox(path: Path, bbox_str: str) -> Georef:
    """Load a plain PNG/JPG with a bbox given as 'west south east north' (WGS84)."""
    parts = [float(x) for x in bbox_str.replace(",", " ").strip().split()]
    if len(parts) != 4:
        raise ValueError("bbox must be a list of 4 numbers: west south east north")
    west, south, east, north = parts
    if not (west < east and south < north):
        raise ValueError(f"invalid bbox: west<east and south<north required, got {parts}")

    rgb = _read_image(path)
    h, w = rgb.shape[:2]
    # Working CRS is WGS84 (degrees) — matches SRTM, so no reprojection is
    # needed for the terrain baseline; ~0.33m GSD is ~3e-6 deg.
    affine = from_bounds(west, south, east, north, w, h)
    return Georef(
        rgb=rgb,
        crs="EPSG:4326",
        transform=affine,
        bbox_wgs84=(west, south, east, north),
        source=path.suffix.lower(),
    )


def load_relative(path: Path) -> Georef:
    """Load a plain image with no georeference (relative-only path)."""
    rgb = _read_image(path)
    return Georef(rgb=rgb, crs=None, transform=None, source=path.suffix.lower())


def from_rgb_array(rgb: np.ndarray) -> Georef:
    """Wrap a raw RGB array as a relative-only Georef (tests, scripts)."""
    return Georef(rgb=np.asarray(rgb, dtype=np.uint8), crs=None, transform=None, source="array")


def load(path: Path | str, bbox: str | None = None) -> Georef:
    """Dispatch to the right loader based on the file + optional bbox."""
    path = Path(path)
    suffix = path.suffix.lower()
    if bbox:
        return load_bbox(path, bbox)
    if suffix in {".tif", ".tiff", ".gtiff"}:
        g = load_georeferenced(path)
        if g.crs is not None:
            return g
    return load_relative(path)
"""SRTM 30m terrain baseline (self-contained, no gdal CLI / API keys).

Fetches SRTM1 (30m, 3601x3601 `Degree tiles` -> GeoTIFFs), stitches the tiles
covering the WGS84 bbox with rasterio.merge, then bilinearly resamples onto the
exact image grid.

Data source: AWS elevation-tiles-prod/skadi (the same source the `elevation`
package uses; reachable without auth, coverage 60N-56S).
"""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import reproject

import depthwizard.config as cfg

_SRTM_NODATA = -32768.0
_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
_SAMPLE = 1.0 / 3600.0  # 30 m
_MAX_TILES = 16


def _tile_name(lat: int, lon: int) -> str:
    """Name of the 1-deg SRTM tile for the cell whose SW corner is (lat, lon)."""
    lat_n = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    # lon names use the *west* edge longitude: W078 covers -78..-77, E005 covers 5..6
    lon_west = lon if lon < 0 else (lon if lon >= 0 else None)
    lon_n = f"W{abs(lon_west):03d}" if lon_west < 0 else f"E{lon_west:03d}"
    return f"{lat_n}{lon_n}"


def _parse_hgt(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=">i2").reshape(3601, 3601).astype(np.float64)
    arr[arr == _SRTM_NODATA] = np.nan
    return arr


def _download(tile: str) -> np.ndarray:
    band = tile[:3]  # e.g. N40
    url = f"{_BASE_URL}/{band}/{tile}.hgt.gz"
    cfg.SRTM_TMP_DIR.mkdir(parents=True, exist_ok=True)
    cache = cfg.SRTM_TMP_DIR / "cache"
    cache.mkdir(exist_ok=True)
    tif_path = cache / f"{tile}.tif"
    if not tif_path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
        data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        arr = _parse_hgt(data)
        lat = int(tile[1:3]) if tile[0] == "N" else -int(tile[1:3])
        lon = -int(tile[4:7]) if tile[3] == "W" else int(tile[4:7])
        if tile[0] == "S":
            lat = -(lat + 1)  # S01 covers -1..0, S15 covers -15..-14
        # affine: tile covers [lat, lat+1) x [lon, lon+1), data starts at top-left
        transform = Affine(_SAMPLE, 0, lon, 0, -_SAMPLE, lat + 1)
        _write_geotiff(tif_path, arr, transform, "EPSG:4326")
    return tif_path


def _write_geotiff(path: Path, arr: np.ndarray, transform: Affine, crs: str) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=rasterio.float64, crs=crs, transform=transform,
                       compress="deflate", nodata=np.nan) as dst:
        dst.write(arr, 1)


def _covering_tiles(west: float, south: float, east: float, north: float) -> list[str]:
    names = []
    for lat in range(int(np.floor(south)), int(np.ceil(north))):
        for lon in range(int(np.floor(west)), int(np.ceil(east))):
            names.append(_tile_name(lat, lon))
    return names


def fetch_srtm(west: float, south: float, east: float, north: float, out_path: Path | None = None) -> Path:
    """Download the SRTM tiles covering the bbox, merge to a GeoTIFF, return path."""
    tiles = _covering_tiles(west, south, east, north)
    if len(tiles) > _MAX_TILES:
        raise RuntimeError(
            f"bbox spans {len(tiles)} SRTM tiles (max {_MAX_TILES}); "
            "tile the input image into smaller areas"
        )

    cfg.SRTM_TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path or (cfg.SRTM_TMP_DIR / f"srtm_{west:.4f}_{south:.4f}_{east:.4f}_{north:.4f}.tif")

    if not out.exists():
        srcs = [_download(t) for t in tiles]
        if len(srcs) == 1:
            out.write_bytes(srcs[0].read_bytes())
        else:
            with rasterio.open(srcs[0]) as f0:
                meta = f0.meta.copy()
            meta.update({"driver": "GTiff", "compress": "deflate", "float64": rasterio.float64,
                         "dtype": "float64", "nodata": np.nan})
            merged, _ = merge(srcs)
            with rasterio.open(out, "w", **meta) as dst:
                dst.write(merged, 1)
    return out


def resample_to_grid(srtm_path: Path, crs: str, transform, shape: tuple) -> np.ndarray:
    """Bilinear resample SRTM (WGS84) onto the image grid -> (H, W) metres."""
    h, w = shape
    dst = np.zeros((h, w), dtype=np.float32)
    with rasterio.open(srtm_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_crs=src.crs,
            dst_crs=crs,
            dst_transform=transform,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=0.0,
        )
    dst[~np.isfinite(dst)] = 0.0
    dst[dst < 0] = 0.0
    return dst


def terrain_baseline(bbox_wgs84: tuple, crs: str, transform, shape: tuple) -> np.ndarray:
    """Fetch + resample the terrain baseline in one call -> (H, W) metres."""
    path = fetch_srtm(*bbox_wgs84)
    return resample_to_grid(path, crs, transform, shape)
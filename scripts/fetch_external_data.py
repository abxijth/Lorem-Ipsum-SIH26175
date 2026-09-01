"""Fetch external (non-GAMUS) landscape validation cases.

For each case (name, landscape, approx bbox, pixels):
  1.  USGS 3DEP Elevation ImageServer export -> bare-earth DEM reference.tif
  2.  Grid-lock: USGS imagery basemap export on the reference's exact bounds
  3.  (optional) Copernicus GLO-30 canopy-surface window clipped to the bounds
      -> surface.tif (used for the forest case to quantify the canopy gap)

Public endpoints, no auth:
  * https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer
  * https://basemap.nationalmap.gov/ArcGIS/rest/services/USGSImageryOnly/MapServer
  * https://copernicus-dem-30m.s3.amazonaws.com  (AWS open data)

All outputs go under data/external/<name>/ with a meta.json recording the
aligned WGS84 bounding box, so the pipeline can be fed the exact same grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import rasterio

DATA = Path(__file__).resolve().parent.parent / "data" / "external"

ELEV_EXPORT = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
IMG_EXPORT = "https://basemap.nationalmap.gov/ArcGIS/rest/services/USGSImageryOnly/MapServer/export"

# Landscape cases: name -> (landscape, approx bbox WGS84 (w,s,e,n), pixels)
CASES = {
    # Ridge-and-valley rural hillside near Asheville, NC (relief ~90 m / 1.4 km).
    "hilly_asheville": ("hilly", (-82.575, 35.49875, -82.56, 35.51375), 1024),
    # Great Smoky Mountains NP, NC (intact canopy; ~600 m relief over 1.4 km).
    "forest_gsmnp": ("forest", (-83.31, 35.59, -83.28, 35.62), 1024),
}


def export(url: str, params: dict, out: Path, timeout: int = 180) -> dict:
    """Call an ArcGIS export returning JSON; download the href to out."""
    params = {**params, "f": "json"}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    meta = r.json()
    if "href" not in meta:
        raise RuntimeError(f"export failed: {meta}")
    img = requests.get(meta["href"], timeout=timeout)
    img.raise_for_status()
    out.write_bytes(img.content)
    return meta


def fetch_case(name: str, landscape: str, bbox: tuple, px: int, force: bool = False) -> None:
    case_dir = DATA / name
    case_dir.mkdir(parents=True, exist_ok=True)
    ref = case_dir / "reference.tif"
    input_img = case_dir / "input.png"
    meta = case_dir / "meta.json"

    if ref.exists() and input_img.exists() and meta.exists() and not force:
        print(f"[{name}] already fetched (use --force to refetch)")
        return

    w, s, e, n = bbox
    print(f"[{name}] exporting 3DEP reference ({px}x{px}) ...", flush=True)
    export(
        ELEV_EXPORT,
        {"bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326,
         "size": f"{px},{px}", "format": "tiff", "pixelType": "F32"},
        ref,
    )

    with rasterio.open(ref) as src:
        b = src.bounds
        extent = [float(b.left), float(b.bottom), float(b.right), float(b.top)]

    print(f"[{name}] exporting imagery grid-locked to {extent} ...", flush=True)
    export(
        IMG_EXPORT,
        {"bbox": ",".join(str(x) for x in extent), "bboxSR": 4326, "imageSR": 4326,
         "size": f"{px},{px}", "format": "png"},
        input_img,
    )

    meta_obj = {
        "name": name,
        "landscape": landscape,
        "bbox": extent,              # exact aligned grid (w, s, e, n)
        "size": px,
        "reference_source": "USGS 3DEP Elevation ImageServer (bare-earth DEM, multi-res)",
        "imagery_source": "USGS ImageryOnly basemap (NAIP-derived ortho)",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta.write_text(json.dumps(meta_obj, indent=2))

    surface_tile = _glo30_tile(extent)
    if surface_tile:
        from rasterio.windows import Window, from_bounds as win_from_bounds
        try:
            with rasterio.open(surface_tile) as src:
                win = win_from_bounds(*extent, transform=src.transform).round_offsets()
                win = win.intersection(Window(0, 0, src.width, src.height))
                arr = src.read(1, window=win)
            sref = case_dir / "surface.tif"
            with rasterio.open(
                sref, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                count=1, dtype="float32", crs="EPSG:4326",
                transform=rasterio.transform.from_bounds(*extent, arr.shape[1], arr.shape[0]),
            ) as dst:
                dst.write(arr.astype(np.float32), 1)
            meta_obj["surface_source"] = f"Copernicus GLO-30 (surface): {surface_tile}"
            meta.write_text(json.dumps(meta_obj, indent=2))
            print(f"[{name}] GLO-30 canopy surface -> surface.tif")
        except Exception as ex:
            print(f"[{name}] GLO-30 fetch failed ({ex!r})", flush=True)
    else:
        print(f"[{name}] no GLO-30 tile covers the bbox; surface.tif skipped")


def _glo30_tile(bbox: list) -> str | None:
    """GLO-30 1x1 deg tiles are named by their south-west corner (N##_W###/E###)."""
    lat = bbox[1]
    lon = bbox[0]
    south = int(np.floor(lat))
    west = int(np.floor(lon))
    # The provider uses the south lat and west lon of each 1x1 deg tile; a bbox
    # whose west edge is -83.x lies in the tile that starts at -84.
    hem = "E" if west >= 0 else "W"
    lon_s = f"{abs(west):03d}"
    name = f"Copernicus_DSM_COG_10_N{south:02d}_00_{hem}{lon_s}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="refetch existing cases")
    ap.add_argument("--cases", help="comma-separated subset of case names")
    args = ap.parse_args()

    names = list(CASES)
    if args.cases:
        names = [c.strip() for c in args.cases.split(",") if c.strip()]
        missing = [c for c in names if c not in CASES]
        if missing:
            sys.exit(f"unknown cases: {missing}; known: {list(CASES)}")

    for name in names:
        landscape, bbox, px = CASES[name]
        fetch_case(name, landscape, bbox, px, force=args.force)


if __name__ == "__main__":
    main()
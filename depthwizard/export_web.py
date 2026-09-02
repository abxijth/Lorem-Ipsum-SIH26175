"""Export processed pipeline output into browser-ready 3D web assets.

Turns the full-resolution DSM grid into a render-resolution height array +
aligned RGB texture + optional colorized error overlay, plus a header.json the
Three.js client reads to scale the mesh, compute slope, and map clicks back to
original image pixel coordinates (for click-to-query height and GCP calibration).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from .georef import Georef
from .dsm import NODATA

# Depth-Anything + fine-tune: structural heights live in a plausible 0-45 m band.
# Keep the mesh/slope rendering in metres when georeferenced.
ERR_CAP_FRACTION = 0.98   # display scale max = p98 of |error| (robust to outliers)
ERR_MIN_RANGE_M = 3.0     # never let the heatmap range collapse below 3 m


def _valid_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr != NODATA)


def _downsample(arr: np.ndarray, dr: int, dc: int, order: int = 1) -> np.ndarray:
    """Bilinear-ish (order 1) downsample to (dr, dc); NaN-safe."""
    if arr.shape[0] == dr and arr.shape[1] == dc:
        return arr.copy()
    sy, sx = dr / arr.shape[0], dc / arr.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = zoom(np.asarray(arr, dtype=np.float64), (sy, sx), order=order, mode="nearest")
    return out


def _texture(rgb: np.ndarray, dr: int, dc: int) -> np.ndarray:
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if (img.height, img.width) != (dr, dc):
        img = img.resize((dc, dr), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _pixel_scale_m(gr: Georef) -> float | None:
    """Ground sampling distance (m) per *original* pixel; None when unknown."""
    if gr.transform is None or gr.crs is None:
        return None
    a = abs(float(gr.transform.a))
    e = abs(float(gr.transform.e))
    if gr.crs == "EPSG:4326" or gr.crs.startswith("EPSG:4326"):
        lat = 0.0
        if gr.bbox_wgs84:
            lat = 0.5 * (gr.bbox_wgs84[1] + gr.bbox_wgs84[3]) * np.pi / 180.0
        c = np.cos(lat)
        a *= 111_320.0 * c
        e *= 111_320.0
    if a <= 0 or e <= 0 or not np.isfinite([a, e]).all():
        return None
    return 0.5 * (a + e)


def _clip_err(err: np.ndarray) -> tuple[np.ndarray, float]:
    valid = _valid_mask(err)
    if not valid.any():
        return np.zeros_like(err), ERR_MIN_RANGE_M
    lo, hi = float(np.nanmin(err[valid])), float(np.nanmax(err[valid]))
    top = float(np.nanpercentile(err[valid].astype(np.float64), ERR_CAP_FRACTION * 100))
    top = max(hi - lo, ERR_MIN_RANGE_M)
    scaled = np.clip(err, 0.0, top)
    return scaled, top


def _colormap_heatmap(err: np.ndarray, top: float) -> np.ndarray:
    """'turbo'-like colorized RGBA from a Float32 error map (nodata transparent)."""
    import matplotlib as mpl
    from matplotlib import colormaps

    mpl.use("Agg")
    cmap = colormaps["turbo"]
    rgba = np.zeros((err.shape[0], err.shape[1], 4), dtype=np.uint8)
    valid = _valid_mask(err) & (err >= 0)
    if not valid.any():
        return rgba
    v = np.clip(err[valid] / top, 0.0, 1.0)
    cols = (cmap(v)[:, :3] * 255.0).astype(np.uint8)
    rgba[valid] = np.concatenate([cols, np.full((cols.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    return rgba


# Deck.gl heightmap: encode height in metres into RGB via the Terrarium scheme.
#   R = floor(h / 256**2 / base) ... we use the standard 3-byte splat in the
#   form assumed by the decoder below (offset applied at decode time).
DECK_MESH_MAX = 1024   # single (non-tiled) mesh grid side limit for Deck.gl
DECK_MESH_MAX_ERROR = 4.0


def terrarium_encode(heights: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Encode Float32 heights (m) into R,G,B so that Deck decodes them back.

    TerrariumRGB: h = offset + (R * 256**2 + G * 256 + B) / 256.  We therefore
    resolve h already shifted + offset-baked into the channel math:
        v = (h - offset) * 256
        R = floor(v / 65536) % 256,  G = floor(v / 256) % 256,  B = floor(v) % 256
    Decoder that reproduces h:  offset + (R*65536 + G*256 + B)/256.
    Range: covers [0, 2^23) m at 1/256 m precision — far beyond our DSM band.
    """
    v = (np.asarray(heights, dtype=np.float64) - offset) * 256.0
    v = np.clip(v, 0.0, 2**23 - 1)
    # Compute each byte as an INTEGER in [0,255] first, then cast to uint8.
    r = np.floor(v / 65536.0) % 256
    g = np.floor(v / 256.0) % 256
    b = np.floor(v) % 256
    return np.stack([r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)], axis=-1)


def terrarium_decode(rgb: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Inverse of :func:`terrarium_encode` — used for round-trip tests."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return offset + (r * 65536.0 + g * 256.0 + b) / 256.0


def export_web_assets(
    gr: Georef,
    dsm: np.ndarray,
    struct: np.ndarray,
    out_dir: Path | str,
    err: np.ndarray | None = None,
    metrics: dict | None = None,
    render_max: int = 1024,
) -> dict:
    """Write heights.bin/struct.bin/tex.jpg/header.json (+err) into out_dir/web/.

    Returns header dict (also mirrored to header.json) for API responses.
    """
    out_dir = Path(out_dir)
    web = out_dir / "web"
    web.mkdir(parents=True, exist_ok=True)

    oh, ow = dsm.shape[:2]
    scale = render_max / max(oh, ow)
    scale = min(1.0, scale)
    rh, rw = max(1, round(oh * scale)), max(1, round(ow * scale))

    dsm = np.asarray(dsm, dtype=np.float32)
    struct = np.asarray(struct, dtype=np.float32)
    h_valid = _valid_mask(dsm)
    s_valid = _valid_mask(struct)

    # Render grid arrays (NaN keeps the client shading, gets cleared for geometry).
    heights = _downsample(dsm, rh, rw)
    structure = _downsample(struct, rh, rw)
    tex = _texture(gr.rgb, rh, rw)

    (web / "heights.bin").write_bytes(heights.astype(np.float32, copy=False).tobytes())
    (web / "struct.bin").write_bytes(structure.astype(np.float32, copy=False).tobytes())
    Image.fromarray(tex, "RGB").save(web / "tex.jpg", quality=88, optimize=True)

    gsd_orig = _pixel_scale_m(gr)
    gsd_m = gsd_orig / scale if gsd_orig else None

    def _range(a: np.ndarray, valid: np.ndarray) -> list:
        vals = a[valid].astype(np.float64)
        return [round(float(vals.min()), 2), round(float(vals.max()), 2)] if vals.size else [0.0, 0.0]

    header = {
        "width": ow,
        "height": oh,
        "grid_w": rw,
        "grid_h": rh,
        "orig_w": ow,
        "orig_h": oh,
        "mode": "absolute" if gr.crs else "relative",
        "crs": gr.crs,
        "bbox": list(gr.bbox_wgs84) if gr.bbox_wgs84 else None,
        "gsd_m": gsd_m,
        "h_range": _range(dsm, h_valid),
        "struct_range": _range(struct, s_valid),
        "has_struct": bool(s_valid.any()),
        "assets": {
            "heights": "web/heights.bin",
            "struct": "web/struct.bin",
            "texture": "web/tex.jpg",
        },
    }

    err_stats = None
    if err is not None and metrics:
        clipped, top = _clip_err(err)
        err_rgba = _downsample(_colormap_heatmap(clipped, top).astype(np.float64), rh, rw)
        Image.fromarray(err_rgba.astype(np.uint8), "RGBA").save(web / "err_heat.png")
        err_stats = {**metrics, "display_max_m": round(float(top), 2)}
        (web / "err_stats.json").write_text(json.dumps(err_stats, indent=2))
        header["assets"]["error"] = "web/err_heat.png"

    # --- Deck.gl TerrainLayer heightmap (TerrariumRGB PNG) ------------------
    # Deck consumes a heightmap *image*, not raw Float32; encode the DSM once at
    # a deck-appropriate mesh size so the browser decodes exact metres back.
    deck_h, deck_w = dsm.shape[:2]
    deck_scale = DECK_MESH_MAX / max(deck_h, deck_w)
    deck_scale = min(1.0, deck_scale)
    deck_h = max(1, round(deck_h * deck_scale))
    deck_w = max(1, round(deck_w * deck_scale))
    deck_dsm = _downsample(dsm, deck_h, deck_w)
    deck_off = 0.0
    deck_rgb = terrarium_encode(deck_dsm, deck_off)
    deck_tex = _texture(gr.rgb, deck_h, deck_w)
    Image.fromarray(deck_rgb, "RGB").save(web / "deck_heights.png")
    Image.fromarray(deck_tex, "RGB").save(web / "deck_tex.jpg", quality=88, optimize=True)
    header["deck"] = {
        "heightsUrl": "web/deck_heights.png",
        "textureUrl": "web/deck_tex.jpg",
        "errorUrl": "web/err_heat.png" if err is not None else None,
        "bounds": list(gr.bbox_wgs84) if gr.bbox_wgs84 else None,
        "elevationDecoder": {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": deck_off},
        "meshMaxError": DECK_MESH_MAX_ERROR,
        "grid": [deck_w, deck_h],
        "mode": "absolute" if gr.crs else "relative",
    }

    (web / "header.json").write_text(json.dumps(header, indent=2))
    return header
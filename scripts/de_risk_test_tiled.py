"""Test whether tile-based (cropped) inference rescues correlation.

Depth-Anything processes images at ~518x518. Feeding a full 1024x1024 nadir
satellite tile may confuse the natural-image prior. This script crops each
tile into overlapping patches (stride = patch_size), runs inference per patch,
stitches the predicted depth back to full resolution, then recomputes
correlation vs. GT nDSM — the plan's prescribed next step for weak correlation.

Usage:
    prime-run python scripts/de_risk_test_tiled.py --tiles 6 --start 0 --patch 518
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
REPO = "earthflow/GAMUS"
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# New York tiles live in train/test with _IMG.h5 image suffix; DC/PHL use val/_RGB.h5.
CITY_SPLITS = {"DC": "val", "PHL": "val", "NYC": "train"}
CITY_IMG_SUFFIX = {"DC": "_RGB.h5", "PHL": "_RGB.h5", "NYC": "_IMG.h5"}


def load_cached(path_in_repo, cache):
    if path_in_repo in cache:
        return cache[path_in_repo]
    p = hf_hub_download(REPO, path_in_repo, repo_type="dataset")
    with h5py.File(p, "r") as f:
        arr = f["image"][...]
    cache[path_in_repo] = arr
    return arr


@torch.inference_mode()
def infer_tiled(img, model, processor, device, patch, stride, tile):
    """Return full-res depth by stitching patch predictions (with feathering)."""
    H, W = img.shape[0], img.shape[1]
    acc = np.zeros((H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)

    def feathered():
        # weight: 1 in patch centre -> 0 at edges, per-pixel
        y = np.hanning(len(np.arange(patch))).astype(np.float64)
        return np.outer(y, y)

    weight = feathered()

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + patch, H)
            x1 = min(x0 + patch, W)
            crop = img[y0:y1, x0:x1]
            d = infer_one(crop, model, processor, device)
            hh, ww = d.shape
            wy = min(y0 + hh, H) - y0
            wx = min(x0 + ww, W) - x0
            acc[y0 : y0 + wy, x0 : x0 + wx] += d[:wy, :wx] * weight[:wy, :wx]
            wsum[y0 : y0 + wy, x0 : x0 + wx] += weight[:wy, :wx]

    nz = wsum > 0
    result = np.zeros((H, W))
    result[nz] = acc[nz] / wsum[nz]
    # fill any gaps (should be none due to overlaps)
    gaps = wsum == 0
    if gaps.any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(gaps, return_distances=False, return_indices=True)
        result[gaps] = result[tuple(idx[i][gaps] for i in range(2))]
    return result


@torch.inference_mode()
def infer_one(crop, model, processor, device):
    img = Image.fromarray(crop)
    inp = processor(images=img, return_tensors="pt").to(device)
    out = model(**inp)
    return out.predicted_depth.squeeze().cpu().numpy()


def corr(d, n):
    d = d.flatten()
    n = n.flatten()
    rp, _ = pearsonr(d, n)
    rs, _ = spearmanr(d, n)
    return rp, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=0, help="optional cap on total tiles across cities (0 = no cap)")
    ap.add_argument("--per-city", type=int, default=4, help="tiles sampled per city")
    ap.add_argument("--cities", default="DC,NYC,PHL", help="comma-separated city prefixes")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--patch", type=int, default=518)
    ap.add_argument("--stride", type=int, default=400)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, model={args.model}, patch={args.patch}, stride={args.stride}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()

    api = HfApi()
    cities = [c.strip().upper() for c in args.cities.split(",")]
    tile_ids = []  # (city, tile_id)
    for c in cities:
        split = CITY_SPLITS.get(c)
        img_suffix = CITY_IMG_SUFFIX.get(c)
        if not split or not img_suffix:
            print(f"  !! no mapping for city {c}, skipping")
            continue
        files = list(api.list_repo_tree(REPO, path_in_repo=f"images/{split}", repo_type="dataset", recursive=False))
        city_tiles = sorted(
            f.path.split("/")[-1][: -len(img_suffix)] for f in files if f.path.split("/")[-1].startswith(c + "_")
        )
        print(f"  {c} ({split}): {len(city_tiles)} tiles available")
        n = min(args.per_city, len(city_tiles))
        take = city_tiles[args.start : args.start + n]
        tile_ids.extend((c, t) for t in take)
        print(f"    sampling {n} tiles from {c}: {take[:3]}...")
    if args.tiles:
        tile_ids = tile_ids[: args.tiles]

    cache = {}
    records = []
    for idx, (city, tid) in enumerate(tile_ids):
        split = CITY_SPLITS[city]
        rgb = load_cached(f"images/{split}/{tid}{CITY_IMG_SUFFIX[city]}", cache)
        ndsm = load_cached(f"heights/{split}/{tid}_AGL.h5", cache).astype(float)
        cls = load_cached(f"classes/{split}/{tid}_CLS.h5", cache)
        depth = infer_tiled(rgb, model, processor, device, args.patch, args.stride, tid)

        H = min(depth.shape[0], ndsm.shape[0])
        W = min(depth.shape[1], ndsm.shape[1])
        dc, nc = depth[:H, :W], ndsm[:H, :W]
        cc = cls[:H, :W]
        rp, rs = corr(dc, nc)

        class_stats = {}
        for c in np.unique(cc[cc >= 0]).astype(int):
            m = cc == c
            if m.sum() < 100:
                continue
            crp, crs = corr(dc[m], nc[m])
            class_stats[int(c)] = {"pearson": float(crp), "spearman": float(crs), "n": int(m.sum())}

        records.append({"tile_id": tid, "city": city, "pearson": float(rp), "spearman": float(rs), "per_class": class_stats})
        print(f"  [{idx}] {city} {tid} pearson={rp:.3f} spearman={rs:.3f}")

    ps = [r["pearson"] for r in records]
    ss = [r["spearman"] for r in records]
    print(f"\nTILED SUMMARY  mean pearson={np.mean(ps):.3f}  mean spearman={np.mean(ss):.3f}")
    for c in cities:
        cr = [r for r in records if r["city"] == c]
        if not cr:
            continue
        cp = [r["pearson"] for r in cr]
        css = [r["spearman"] for r in cr]
        print(f"  [{c}] n={len(cp)} pearson mean={np.mean(cp):.3f} (min {min(cp):.3f}, max {max(cp):.3f}) "
              f"| spearman mean={np.mean(css):.3f}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.joinpath("de_risk_tiled_results.json").write_text(
        json.dumps({"model": args.model, "patch": args.patch, "stride": args.stride, "tiles": records}, indent=2)
    )


if __name__ == "__main__":
    main()

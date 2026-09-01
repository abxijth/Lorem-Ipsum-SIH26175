"""Phase 0 — De-risk test (GAMUS).

Validates the core assumption: does Depth-Anything V2 (trained on natural
images) produce usable relative-depth output on nadir (top-down) satellite
images from the GAMUS dataset?

GAMUS structure (HF earthflow/GAMUS, imagefolder of HDF5 per tile):
    images/  <split>/  <ID>_RGB.h5   -> 'image' (1024,1024,3) uint8
    heights/ <split>/  <ID>_AGL.h5   -> 'image' (1024,1024) float32   (nDSM, metres)
    classes/ <split>/  <ID>_CLS.h5   -> 'image' (1024,1024) float32   (semantic labels)

Each tile is 1024x1024 at 0.33 m ground sampling distance.

Pipeline per tile:
    RGB -> Depth-Anything V2 -> relative depth
    compare relative depth vs. ground-truth nDSM (AGL)
    via Pearson r and Spearman (rank) rho.

Usage:
    prime-run python scripts/de_risk_test.py [--tiles N] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

REPO = "earthflow/GAMUS"
SUFFIX = {"images": "_RGB.h5", "heights": "_AGL.h5", "classes": "_CLS.h5"}
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def normalize01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def load_tile(api: HfApi, tile_id: str, split: str, cache: dict) -> dict:
    """Download + cache one tile's rgb/ndsm/classes, return as arrays."""
    out = {}
    for key, suffix in SUFFIX.items():
        path_in_repo = f"{key}/{split}/{tile_id}{suffix}"
        if path_in_repo in cache:
            out[key] = cache[path_in_repo]
            continue
        p = hf_hub_download(REPO, path_in_repo, repo_type="dataset")
        with h5py.File(p, "r") as f:
            arr = f["image"][...]
        cache[path_in_repo] = arr
        out[key] = arr
    return out


def correlation_stats(depth: np.ndarray, ndsm: np.ndarray):
    d = depth.flatten()
    n = ndsm.flatten()
    r_p, p_p = pearsonr(d, n)
    r_s, p_s = spearmanr(d, n)
    r_norm, _ = pearsonr(normalize01(d), normalize01(n))
    return {
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "spearman_rho": float(r_s),
        "spearman_p": float(p_s),
        "pearson_norm_r": float(r_norm),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--split", default="val")
    ap.add_argument("--start", type=int, default=0, help="skip first N tiles")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print(f"Loading model & processor: {args.model}")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device)
    model.eval()
    print(f"  model loaded in {time.time() - t0:.1f}s")

    print(f"Listing GAMUS tiles in '{args.split}' split...")
    api = HfApi()
    files = list(
        api.list_repo_tree(REPO, path_in_repo=f"images/{args.split}", repo_type="dataset", recursive=False)
    )
    tile_ids = sorted(f.path.split("/")[-1][: -len(SUFFIX["images"])] for f in files)
    tile_ids = tile_ids[args.start : args.start + args.tiles]
    print(f"  using {len(tile_ids)} tiles: {tile_ids[:5]}...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    timings = []
    cache = {}

    for idx, tile_id in enumerate(tile_ids):
        t0 = time.time()
        tile = load_tile(api, tile_id, args.split, cache)
        rgb = np.asarray(tile["images"])  # (H,W,3) uint8
        ndsm = np.asarray(tile["heights"], dtype=float)  # (H,W) nDSM metres
        cls = np.asarray(tile["classes"])

        img = Image.fromarray(rgb)
        t_inf = time.time()
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        depth = out.predicted_depth.squeeze().cpu().numpy()
        timings.append(time.time() - t_inf)

        # align shapes (depth may differ in resolution)
        H = min(depth.shape[0], ndsm.shape[0])
        W = min(depth.shape[1], ndsm.shape[1])
        depth_c = depth[:H, :W]
        ndsm_c = ndsm[:H, :W]
        cls_c = cls[:H, :W]

        stats = correlation_stats(depth_c, ndsm_c)
        stats["tile_id"] = tile_id
        stats["shape"] = [H, W]

        # Per-class correlation (where valid, ndsm>0 and sensible)
        class_ids = np.unique(cls_c[cls_c >= 0]).astype(int)
        class_stats = {}
        for c in class_ids:
            m = cls_c == c
            if m.sum() < 100:
                continue
            cs = correlation_stats(depth_c[m], ndsm_c[m])
            class_stats[int(c)] = {
                "n": int(m.sum()),
                "pearson_r": cs["pearson_r"],
                "spearman_rho": cs["spearman_rho"],
            }
        stats["per_class"] = class_stats

        records.append(stats)
        print(
            f"  [{idx}] {tile_id}  pearson r={stats['pearson_r']:.3f}  "
            f"spearman rho={stats['spearman_rho']:.3f}  "
            f"computed in {time.time() - t0:.1f}s"
        )

    print("\n======== SUMMARY ========")
    if records:
        ps = [r["pearson_r"] for r in records]
        ss = [r["spearman_rho"] for r in records]
        print(f"Pearson r:   mean={np.mean(ps):.3f}  median={np.median(ps):.3f}  "
              f"min={min(ps):.3f}  max={max(ps):.3f}")
        print(f"Spearman rho: mean={np.mean(ss):.3f}  median={np.median(ss):.3f}  "
              f"min={min(ss):.3f}  max={max(ss):.3f}")
        if timings:
            print(f"Avg inference time/tile: {np.mean(timings):.2f}s")

    out_path = RESULTS_DIR / "de_risk_results.json"
    out_path.write_text(json.dumps({"model": args.model, "split": args.split, "tiles": records}, indent=2))
    print(f"\nSaved results to {out_path}")

    print("\nVerdict:")
    mean_r = float(np.mean(ps)) if records else 0.0
    if mean_r > 0.6:
        verdict = "PROCEED — strong correlation. Calibration is a refinement problem."
    elif mean_r > 0.3:
        verdict = "WEAK-USABLE — correlation present but noisy. Lean harder on the calibration head (Day 2)."
    else:
        verdict = "PIVOT — weak correlation. Consider tile-based inference, shading blend, or narrower demo scope."
    print(f"  {verdict}")


if __name__ == "__main__":
    main()

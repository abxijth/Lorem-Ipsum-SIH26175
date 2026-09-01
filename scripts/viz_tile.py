"""Produce side-by-side visualisations of RGB | predicted depth | GT nDSM.

Used to characterise the de-risk failure mode qualitatively for documentation.
"""

from __future__ import annotations

import argparse

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

REPO = "earthflow/GAMUS"
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def load(path_in_repo):
    p = hf_hub_download(REPO, path_in_repo, repo_type="dataset")
    with h5py.File(p, "r") as f:
        return f["image"][...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="DC_02_26")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

    rgb = load(f"images/{args.split}/{args.tile}_RGB.h5")
    ndsm = load(f"heights/{args.split}/{args.tile}_AGL.h5")
    cls = load(f"classes/{args.split}/{args.tile}_CLS.h5")
    print("rgb", rgb.shape, "ndsm", ndsm.shape, "cls", cls.shape)

    img = Image.fromarray(rgb)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    depth = out.predicted_depth.squeeze().cpu().numpy()

    H = min(depth.shape[0], ndsm.shape[0])
    W = min(depth.shape[1], ndsm.shape[1])
    depth = depth[:H, :W]
    ndsm = ndsm[:H, :W]
    cls = cls[:H, :W]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB (GAMUS {args.tile})")
    axes[0, 1].imshow(ndsm, cmap="terrain")
    axes[0, 1].set_title(f"GT nDSM (AGL)  min={ndsm.min():.1f} max={ndsm.max():.1f} m")
    axes[1, 0].imshow(depth, cmap="viridis")
    axes[1, 0].set_title(f"Depth-Anything relative depth  min={depth.min():.2f} max={depth.max():.2f}")
    axes[1, 1].imshow(cls, cmap="tab10", vmin=-1, vmax=10)
    axes[1, 1].set_title("GAMUS semantic classes")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()

    out = args.out or "results/viz.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

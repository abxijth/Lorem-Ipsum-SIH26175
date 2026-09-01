"""Sparse fine-tune of Depth-Anything-V2-Small on GAMUS for metric structural height.

Why (honest framing): the shipped baseline keeps DA-V2 frozen and fits a single
GLOBAL linear calibration of relative depth -> nDSM. That calibration fails where
the relative-depth signal is entangled with terrain relief, so held-out DC/NYC
hit Pearson ~0.1. This fixture trains the model to predict nDSM (AGL, metres)
directly, getting the "backbone trained on GAMUS" half of the problem statement
and attacking the structural-signal failure at the source.

Design (tractable on a laptop GPU):
  - freeze lowest 10/12 encoder layers (keep texture), train last 2 + neck + head
  - regress predicted_depth onto AGL nDSM (scaled to ~0..1) with SmoothL1
  - patch-based (518x518) crops from fresh GAMUS train tiles (held-out excluded)
  - persist state_dict; loader resolves it directly

Usage:
    prime-run ./.venv/bin/python scripts/finetune_gamus.py \
        --train-per-city 6 --epochs 3 --patches-per-tile 16 --workers 0 --out models/finetuned-gamus
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _gamus  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
PATCH = 518
NORM = 45.0  # AGL nDSM metres -> ~[0,1]; model predicts scaled value, we multiply back


def build_model():
    m = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    # freeze everything, then unfreeze the parts we want
    for p in m.parameters():
        p.requires_grad = False
    for name, p in m.named_parameters():
        # last two encoder layers + neck (decoder reassembly) + regression head
        if any(
            name.startswith(prefix)
            for prefix in ("encoder.layer.10", "encoder.layer.11", "neck.", "head.")
        ):
            p.requires_grad = True
    return m


def trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_crops(rgb: np.ndarray, n: int, rng: random.Random) -> list[tuple[int, int]]:
    """Random 518x518 top-left positions inside the 1024x1024 tile."""
    h, w = rgb.shape[:2]
    positions = []
    for _ in range(n):
        y0 = rng.randint(0, h - PATCH)
        x0 = rng.randint(0, w - PATCH)
        positions.append((y0, x0))
    return positions


class GamusDataset(Dataset):
    def __init__(self, tiles: list[tuple[str, str]], patches_per_tile: int, seed: int = 0):
        self.tiles = tiles
        self.ppt = patches_per_tile
        self.rng = random.Random(seed)
        # (city, tid, y0, x0)
        self.items: list[tuple[str, str, int, int]] = []
        for city, tid in tiles:
            rgb = _gamus.tile_rgb(city, "train", tid)
            for y0, x0 in make_crops(rgb, patches_per_tile, self.rng):
                self.items.append((city, tid, y0, x0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        city, tid, y0, x0 = self.items[i]
        rgb = _gamus.tile_rgb(city, "train", tid)[y0 : y0 + PATCH, x0 : x0 + PATCH]
        ndsm = _gamus.tile_ndsm(city, "train", tid)[y0 : y0 + PATCH, x0 : x0 + PATCH]
        rgb = np.ascontiguousarray(rgb).astype(np.uint8)  # NHWC, matched to inference
        target = torch.from_numpy(ndsm.astype(np.float32)) / NORM
        return rgb, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-city", type=int, default=6)
    ap.add_argument("--cities", default="DC,PHL,NYC")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patches-per-tile", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/finetuned-gamus")
    args = ap.parse_args()

    heldout_ids = {tid for _, tid in _gamus.heldout_entries()}
    train_tiles = [
        (c.strip().upper(), t)
        for c in args.cities.split(",")
        for t in _gamus.train_tiles(c.strip().upper(), exclude=heldout_ids, limit=args.train_per_city)
    ]
    print(f"train tiles: {len(train_tiles)} {train_tiles[:6]}...", flush=True)

    ds = GamusDataset(train_tiles, args.patches_per_tile, seed=args.seed)
    # DA-V2 processor applies ImageNet mean/std + resize: MUST match _infer_patch exactly.
    # Build a list-based batch through the processor for distribution parity.
    def collate(items):
        rbs, tgts = zip(*items)
        inp = proc(images=list(rbs), return_tensors="pt")
        return inp, torch.stack(tgts)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=0, pin_memory=True, drop_last=False)
    print(f"patches: {len(ds)} (batches: {len(loader)})", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = build_model().to(dev)
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    print(f"model on {dev}; trainable params: {trainable_params(model)/1e6:.2f} M", flush=True)

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    other_train = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    opt = torch.optim.AdamW(
        [{"params": other_train, "lr": args.lr}, {"params": head_params, "lr": args.lr * 4}]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    model.train()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tot, n = 0.0, 0
        for bi, (inp, target) in enumerate(loader):
            rgb = inp["pixel_values"].to(dev)
            target = target.to(dev)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                pred = model(pixel_values=rgb.half() if dev == "cuda" else rgb).predicted_depth  # (B, H, W)
                loss = F.smooth_l1_loss(pred, target)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * rgb.size(0)
            n += rgb.size(0)
            if bi % 5 == 0:
                print(f"  ep{ep} b{bi} loss={loss.item():.4f}", flush=True)
        print(f"epoch {ep}/{args.epochs} loss={tot/n:.4f} ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    import json
    meta = {"norm": NORM, "model_id": MODEL_ID, "train_tiles": train_tiles,
            "epochs": args.epochs, "patches_per_tile": args.patches_per_tile}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"fine-tuned weights -> {out / 'pytorch_model.bin'}")


if __name__ == "__main__":
    main()

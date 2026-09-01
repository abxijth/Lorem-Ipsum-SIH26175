"""Central configuration for the DepthWizard pipeline."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Depth model
# ---------------------------------------------------------------------------
BACKEND = "depthwizard"

# Locked from Phase 0 de-risk: Small is sufficient, Base buys little.
MODEL_ID = os.environ.get("DW_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")

# Fine-tuned backbone (GAMUS -> structural nDSM prior). Default production path.
# Set DW_FINETUNED="off" to fall back to the stock frozen Depth-Anything weights
# (used by the baseline A/B and to reproduce the original numbers).
FINETUNED_DIR = Path(os.environ.get("DW_FINETUNED", str(ROOT / "models" / "finetuned")))

# Depth-Anything native resolution & tiling parameters (validated on GAMUS).
PATCH_SIZE = int(os.environ.get("DW_PATCH", "518"))
STRIDE = int(os.environ.get("DW_STRIDE", "400"))

# Very large inputs are downscaled for inference if they exceed this side length.
MAX_INFER_DIM = int(os.environ.get("DW_MAX_DIM", "4096"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

CALIBRATION_PATH = os.environ.get("DW_CALIBRATION", str(ROOT / "models" / "calibration.json"))

# Per-class calibration (calibration v2) + inference-time segmentation head.
CALIBRATION_V2_PATH = os.environ.get("DW_CALIBRATION_V2", str(ROOT / "models" / "calibration_v2.json"))
SEG_WEIGHTS_PATH = os.environ.get("DW_SEG", str(ROOT / "models" / "gamus_seg.pt"))
SEG_PATCH = 512
SEG_STRIDE = int(os.environ.get("DW_SEG_STRIDE", "448"))

# SRTM (temporary working dir for eio tile downloads)
SRTM_TMP_DIR = DATA_DIR / "srtm"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
# Resolved to "cuda"/"cpu" at model load time (see depth.py). Explicitly
# override with DW_DEVICE if needed (e.g. "cpu"). On this Optimus laptop the
# whole process must be launched via prime-run to reach the dGPU.
DEVICE = os.environ.get("DW_DEVICE", "auto")
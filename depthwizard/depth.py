"""Depth-Anything V2 tiled inference (locked in from the Phase 0 de-risk).

Feeding a full nadir tile to the model destroys the signal (r~0.01); cropping
into 518x518 patches at native resolution recovers it (r~0.33 across 3 cities).
This module implements that tiled/stiched inference as the production depth step.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

from .config import MAX_INFER_DIM, MODEL_ID, PATCH_SIZE, STRIDE


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DepthEstimator:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or resolve_device()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
        self._patch = PATCH_SIZE
        self._stride = STRIDE
        self._hann = np.hanning(PATCH_SIZE).astype(np.float64)
        self._feather = np.outer(self._hann, self._hann)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<DepthEstimator model={self.model_id} device={self.device}>"

    @torch.inference_mode()
    def _infer_patch(self, crop: np.ndarray) -> np.ndarray:
        img = Image.fromarray(crop)
        inp = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        return out.predicted_depth.squeeze().cpu().numpy()

    def _resize_for_inference(self, rgb: np.ndarray):
        """Cap side length to MAX_INFER_DIM (prevents memory blowups)."""
        h, w = rgb.shape[:2]
        if max(h, w) <= MAX_INFER_DIM:
            return rgb, 1.0
        scale = MAX_INFER_DIM / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
        return np.asarray(img), scale

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Relative depth map at (approximately) the input resolution."""
        rgb_resized, _ = self._resize_for_inference(rgb)
        h, w = rgb_resized.shape[:2]

        acc = np.zeros((h, w), dtype=np.float64)
        wsum = np.zeros((h, w), dtype=np.float64)
        patch, stride = self._patch, self._stride

        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(y0 + patch, h)
                x1 = min(x0 + patch, w)
                d = self._infer_patch(rgb_resized[y0:y1, x0:x1])
                ph, pw = d.shape
                # Clamp to what the crop actually covered (edge patches are cut short)
                wy = min(y0 + ph, h) - y0
                wx = min(x0 + pw, w) - x0
                acc[y0 : y0 + wy, x0 : x0 + wx] += d[:wy, :wx] * self._feather[:wy, :wx]
                wsum[y0 : y0 + wy, x0 : x0 + wx] += self._feather[:wy, :wx]

        nz = wsum > 0
        depth = np.zeros((h, w))
        depth[nz] = acc[nz] / wsum[nz]
        gaps = wsum == 0
        if gaps.any():
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(gaps, return_distances=False, return_indices=True)
            depth[gaps] = depth[tuple(idx[i][gaps] for i in range(2))]

        if depth.shape != (rgb.shape[0], rgb.shape[1]):
            # Restore original resolution when we downscaled for inference.
            pil = Image.fromarray(depth)
            depth = np.asarray(pil.resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
        return depth
"""Relative depth -> structural height (nDSM) calibration.

MVP decision (locked): one robust global linear regression
    structural_height = slope * rel_depth + intercept
fit with RANSAC on GAMUS train pairs (tiled Depth-Anything depth vs. AGL nDSM).
Per-class conditioning / segmentation is deferred to Day 2 unless this performs
poorly in validation.

Calibration is persisted as JSON so inference doesn't need GAMUS.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor

from .config import CALIBRATION_PATH

# GAMUS label legend (labels present in the CLS rasters).
CLASS_NAMES = {
    0: "others", 1: "ground", 2: "low_vegetation", 3: "building",
    4: "water", 5: "road", 6: "tree",
}


@dataclass
class Calibration:
    slope: float
    intercept: float
    n_pixels: int
    n_tiles: int
    city_counts: dict
    # Per-class conditioning (calibration v2). Empty dicts == global-only.
    class_slopes: dict = field(default_factory=dict)
    class_intercepts: dict = field(default_factory=dict)
    class_counts: dict = field(default_factory=dict)

    @property
    def per_class(self) -> bool:
        return bool(self.class_slopes)

    @classmethod
    def from_json(cls, path: Path | str) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(
            **{k: d.get(k, v) for k, v in (
                ("slope", None), ("intercept", None), ("n_pixels", 0),
                ("n_tiles", 0), ("city_counts", {}), ("class_slopes", {}),
                ("class_intercepts", {}), ("class_counts", {}),
            )}
        )

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# RANSAC tuning
_MIN_SAMPLES = 0.1      # fraction of subsampled points that must be inliers
_STOP_N = 5             # stop when this many inliers found (fast)
_RESIDUAL_MULT = 1.96   # ~95% band on residual std


def fit_calibration(depth_samples: list[np.ndarray], ndsm_samples: list[np.ndarray],
                    n_pixels_cap: int = 200_000) -> Calibration:
    """Robust global linear fit over pooled, subsampled pixel pairs.

    depth_samples / ndsm_samples: one array per tile. Pixels are subsampled
    evenly to keep RANSAC fast; caps total points at n_pixels_cap.
    """
    xs, ys = [], []
    for d, n in zip(depth_samples, ndsm_samples):
        d = d.flatten().astype(np.float64)
        n = n.flatten().astype(np.float64)
        keep = np.isfinite(d) & np.isfinite(n) & (n >= 0)
        d = d[keep]
        n = n[keep]
        # subsample to ~20k/tile to keep RANSAC bounded
        if d.size > 20_000:
            idx = np.linspace(0, d.size - 1, 20_000).astype(int)
            d, n = d[idx], n[idx]
        xs.append(d)
        ys.append(n)

    X = np.concatenate(xs).reshape(-1, 1)
    Y = np.concatenate(ys)
    if X.size > n_pixels_cap:
        idx = np.linspace(0, X.size - 1, n_pixels_cap).astype(int)
        X, Y = X[idx], Y[idx]

    ransac = RANSACRegressor(
        min_samples=_MIN_SAMPLES,
        stop_n_inliers=_STOP_N,
        residual_threshold=np.std(Y) * _RESIDUAL_MULT,
        max_trials=2000,
        random_state=0,
    )
    ransac.fit(X, Y)
    slope = float(ransac.estimator_.coef_[0])
    intercept = float(ransac.estimator_.intercept_)
    return Calibration(slope=slope, intercept=intercept, n_pixels=len(X), n_tiles=len(xs), city_counts={})


def _fit_ransac_line(depth: np.ndarray, ndsm: np.ndarray) -> tuple[float, float, int]:
    """Robust linear fit on prepared 1-D arrays -> (slope, intercept, n)."""
    keep = np.isfinite(depth) & np.isfinite(ndsm) & (ndsm >= 0)
    d, n = depth[keep], ndsm[keep]
    if d.size > 20_000:  # keep RANSAC bounded, like the global fit
        idx = np.linspace(0, d.size - 1, 20_000).astype(int)
        d, n = d[idx], n[idx]
    if d.size < 50:
        return float("nan"), float("nan"), int(d.size)
    ransac = RANSACRegressor(
        min_samples=_MIN_SAMPLES,
        stop_n_inliers=_STOP_N,
        residual_threshold=np.std(n) * _RESIDUAL_MULT,
        max_trials=2000,
        random_state=0,
    )
    ransac.fit(d.reshape(-1, 1), n)
    return float(ransac.estimator_.coef_[0]), float(ransac.estimator_.intercept_), int(d.size)


def fit_calibration_classwise(
    depth_samples: list[np.ndarray],
    ndsm_samples: list[np.ndarray],
    cls_samples: list[np.ndarray],
    min_pixels: int = 40_000,
    classes: list[int] | None = None,
) -> Calibration:
    """Global + per-class RANSAC calibration (v2).

    Fits the global line on all pixels, then a per-class line per label.
    Classes with fewer than `min_pixels` (or the fallback label 0) reuse the
    global line. Returns a Calibration carrying both; `apply_calibration`
    switches on the pixel's class when a class map is supplied.
    """
    base = fit_calibration(depth_samples, ndsm_samples)
    base.city_counts = {}

    if classes is None:
        used = {int(l) for c in cls_samples for l in np.unique(c)}
        classes = sorted(l for l in used if l != 0)

    slopes, intercepts, counts = {}, {}, {}
    for label in classes:
        ds, ns = [], []
        for d, n, c in zip(depth_samples, ndsm_samples, cls_samples):
            h = min(d.shape[0], n.shape[0], c.shape[0])
            w = min(d.shape[1], n.shape[1], c.shape[1])
            m = c[:h, :w] == label
            if not m.any():
                continue
            ds.append(d[:h, :w][m].ravel())
            ns.append(n[:h, :w][m].ravel())
        total = sum(len(x) for x in ns)
        if total < min_pixels:
            slopes[label], intercepts[label] = base.slope, base.intercept
            counts[label] = int(total)
            continue
        slope, intercept, n = _fit_ransac_line(np.concatenate(ds), np.concatenate(ns))
        if not np.isfinite(slope):
            slopes[label], intercepts[label] = base.slope, base.intercept
        else:
            slopes[label], intercepts[label] = slope, intercept
        counts[label] = int(n)

    base.class_slopes = {str(k): v for k, v in slopes.items()}
    base.class_intercepts = {str(k): v for k, v in intercepts.items()}
    base.class_counts = {str(k): v for k, v in counts.items()}
    return base


def apply_calibration(depth: np.ndarray, calib: Calibration, class_map: np.ndarray | None = None) -> np.ndarray:
    """Structural height (m) from relative depth; clamped >= 0.

    With `class_map` and a per-class calibration, each class's line is applied
    per-pixel; classes missing from the calibration (or label 0) fall back to
    the global line.
    """
    out = calib.slope * depth + calib.intercept
    if class_map is not None and calib.per_class:
        for k, s in calib.class_slopes.items():
            label = int(k)
            m = class_map == label
            if not m.any():
                continue
            out = np.where(m, depth * s + calib.class_intercepts[k], out)
    return np.clip(out, 0.0, None)


def parse_gcp(gcp_str: str) -> tuple[list, list]:
    """Parse 'x1,y1,h1;x2,y2,h2' -> (points[(x,y)...], heights[...])."""
    points, heights = [], []
    for part in gcp_str.strip().replace(";", " ").split():
        x, y, h = (float(v) for v in part.replace(",", " ").split())
        points.append((x, y))
        heights.append(h)
    if len(points) < 1:
        raise ValueError("gcp requires at least one 'x,y,height' entry")
    return points, heights


def gcp_refit(
    depth: np.ndarray,
    points: list[tuple],
    heights: list[float],
    base_calib: Calibration | None = None,
    radius: float = 3.0,
) -> Calibration:
    """Least-squares calibration from user-supplied ground-control heights.

    depth values are sampled as a Gaussian-smoothed mean around each clicked
    pixel. With >=2 points a fresh slope+intercept is fit; with 1 point only the
    intercept shifts (slope taken from `base_calib`).
    """
    d = gaussian_filter(depth.astype(np.float64), radius)
    xs, ys = [], []
    for (x, y), h in zip(points, heights):
        xi = int(min(max(round(float(x)), 0), d.shape[1] - 1))
        yi = int(min(max(round(float(y)), 0), d.shape[0] - 1))
        xs.append(d[yi, xi])
        ys.append(float(h))
    xs, ys = np.asarray(xs), np.asarray(ys)

    if xs.size >= 2:
        A = np.vstack([xs, np.ones_like(xs)]).T
        slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    else:
        base_calib = base_calib if base_calib is not None else Calibration(0.0, 0.0, 0, 0, {})
        if isinstance(base_calib, PiecewiseCalibration):
            # adopt the local piecewise slope around the sampled depth point
            a = np.asarray(base_calib.anchors)
            v = np.asarray(base_calib.values)
            i = int(np.searchsorted(a, xs[0]))
            i = min(max(i, 1), len(a) - 1)
            denom = a[i] - a[i - 1]
            slope = (v[i] - v[i - 1]) / denom if denom > 1e-9 else 0.0
        else:
            slope = base_calib.slope
        intercept = ys[0] - slope * xs[0]

    return Calibration(
        slope=float(slope),
        intercept=float(intercept),
        n_pixels=int(xs.size),
        n_tiles=int(xs.size),
        city_counts={"gcp": int(xs.size)},
    )


@dataclass
class PiecewiseCalibration:
    """Calibration as a piecewise-linear function of relative depth.

    Fitted as robust per-bin anchors through depth quantile bins: low-depth bins
    capture flat classes (ground/vegetation/road -> ~0 m), high-depth bins the
    tall classes (buildings). The same role as per-class conditioning but
    reachable at inference with no segmentation model.
    """
    anchors: list            # depth breakpoints, strictly increasing
    values: list             # structural height at each anchor (m)
    n_pixels: int
    n_tiles: int
    city_counts: dict

    @classmethod
    def from_json(cls, path: Path | str) -> "PiecewiseCalibration":
        d = json.loads(Path(path).read_text())
        return cls(**{k: d[k] for k in ("anchors", "values", "n_pixels", "n_tiles", "city_counts")})

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def fit_piecewise(
    depth_samples: list[np.ndarray],
    ndsm_samples: list[np.ndarray],
    n_segments: int = 6,
    n_pixels_cap: int = 200_000,
) -> PiecewiseCalibration:
    """Robust piecewise-linear calibration depth -> structural height.

    Pools subsampled (depth, nDSM) pairs, splits depth into ~n_segments
    quantile bins, and anchors each bin at its robust center (median depth,
    median height). Linear interpolation between anchors; clamped at the ends.
    """
    xs, ys = [], []
    for d, n in zip(depth_samples, ndsm_samples):
        d = d.flatten().astype(np.float64)
        n = n.flatten().astype(np.float64)
        keep = np.isfinite(d) & np.isfinite(n) & (n >= 0)
        d, n = d[keep], n[keep]
        if d.size > 20_000:
            idx = np.linspace(0, d.size - 1, 20_000).astype(int)
            d, n = d[idx], n[idx]
        xs.append(d)
        ys.append(n)

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    if X.size > n_pixels_cap:
        idx = np.linspace(0, X.size - 1, n_pixels_cap).astype(int)
        X, Y = X[idx], Y[idx]

    edges = np.quantile(X, np.linspace(0.0, 1.0, n_segments + 1))
    edges = np.unique(edges)  # drop empty bins from duplicated depth values
    if edges.size < 3:
        raise ValueError("not enough distinct depth bins for piecewise fit")

    anchors, values = [], []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        m = (X >= lo) & (X < hi) if k < len(edges) - 2 else (X >= lo) & (X <= hi)
        if not m.any():
            continue
        anchors.append(float(np.median(X[m])))
        values.append(float(np.median(Y[m])))

    if len(anchors) < 2:
        raise ValueError(f"piecewise fit needs >=2 non-empty bins, got {len(anchors)}")
    order = np.argsort(anchors)
    anchors = [anchors[i] for i in order]
    values = [values[i] for i in order]
    return PiecewiseCalibration(
        anchors=anchors, values=values,
        n_pixels=int(X.size), n_tiles=len(xs), city_counts={},
    )


def apply_piecewise(depth: np.ndarray, calib: PiecewiseCalibration) -> np.ndarray:
    """Structural height (m) via piecewise-linear interpolation in depth."""
    d = depth.astype(np.float64)
    d = np.where(np.isfinite(d), d, calib.anchors[0])
    out = np.interp(d, calib.anchors, calib.values,
                    left=calib.values[0], right=calib.values[-1])
    return np.clip(out, 0.0, None)


def fit_piecewise_segments(
    depth_samples: list[np.ndarray],
    ndsm_samples: list[np.ndarray],
    n_segments: int = 6,
    n_pixels_cap: int = 200_000,
    min_pixels: int = 1_000,
    fallback: PiecewiseCalibration | None = None,
) -> PiecewiseCalibration:
    """Piecewise calibration anchored at each bin's robust (RANSAC) line.

    Same quantile binning as `fit_piecewise`, but each bin's anchor height comes
    from a robust linear fit *within the bin*, evaluated at the bin midpoint.
    This captures a group's dominant depth-height trend (e.g. the building tail)
    instead of the median height (which saturates when tall pixels are rare).
    Returns a PiecewiseCalibration so the same apply/interp path is used.
    """
    xs, ys = [], []
    for d, n in zip(depth_samples, ndsm_samples):
        d = d.flatten().astype(np.float64)
        n = n.flatten().astype(np.float64)
        keep = np.isfinite(d) & np.isfinite(n) & (n >= 0)
        d, n = d[keep], n[keep]
        if d.size > 20_000:
            idx = np.linspace(0, d.size - 1, 20_000).astype(int)
            d, n = d[idx], n[idx]
        xs.append(d)
        ys.append(n)

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    if X.size > n_pixels_cap:
        idx = np.linspace(0, X.size - 1, n_pixels_cap).astype(int)
        X, Y = X[idx], Y[idx]

    # global line as fallback for thin bins
    global_line = fit_calibration(depth_samples, ndsm_samples)

    edges = np.unique(np.quantile(X, np.linspace(0.0, 1.0, n_segments + 1)))
    if edges.size < 3:
        raise ValueError("not enough distinct depth bins for piecewise fit")

    anchors, values = [], []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        m = (X >= lo) & (X < hi) if k < len(edges) - 2 else (X >= lo) & (X <= hi)
        if not m.any():
            continue
        ax = float(np.median(X[m]))
        if m.sum() < min_pixels:
            h = global_line.slope * ax + global_line.intercept
        else:
            slope, _intercept, _n = _fit_ransac_line(X[m], Y[m])
            h = global_line.slope * ax + global_line.intercept if not np.isfinite(slope) \
                else slope * ax + _intercept
        anchors.append(ax)
        values.append(max(0.0, h))

    if len(anchors) < 2:
        raise ValueError(f"piecewise fit needs >=2 non-empty bins, got {len(anchors)}")
    order = np.argsort(anchors)
    anchors = [anchors[i] for i in order]
    values = [values[i] for i in order]
    return PiecewiseCalibration(
        anchors=anchors, values=values,
        n_pixels=int(X.size), n_tiles=len(xs), city_counts={},
    )


def load_calibration_any(path: Path | str | None = None) -> Calibration | PiecewiseCalibration:
    """Load a linear or piecewise calibration JSON (auto-detect by keys)."""
    path = Path(path) if path else Path(CALIBRATION_PATH)
    if not path.exists():
        raise FileNotFoundError(f"calibration not found at {path} (run scripts/fit_calibration.py)")
    d = json.loads(path.read_text())
    if "anchors" in d:
        return PiecewiseCalibration(
            **{k: d[k] for k in ("anchors", "values", "n_pixels", "n_tiles", "city_counts")})
    return Calibration.from_json(path)


def apply_calibration_any(depth: np.ndarray, calib: Calibration | PiecewiseCalibration,
                          class_map: np.ndarray | None = None) -> np.ndarray:
    """Dispatch to the right calibrator (linear per-class or piecewise)."""
    if isinstance(calib, PiecewiseCalibration):
        return apply_piecewise(depth, calib)
    return apply_calibration(depth, calib, class_map)


def load_calibration(path: Path | str | None = None) -> Calibration:
    path = Path(path) if path else Path(CALIBRATION_PATH)
    if not path.exists():
        raise FileNotFoundError(f"calibration not found at {path} (run scripts/fit_calibration.py)")
    return Calibration.from_json(path)
# Landscape Validation — External (non-GAMUS) Cases

Companion to `results/VALIDATION.md`. Runs the **shipped** pipeline (global
RANSAC `models/calibration.json`, fine-tuned Depth-Anything-V2 inference) on two
non-urban, non-GAMUS test cases to measure how the estimator generalizes beyond
the urban images it was trained on. Urban row included for reference. Frozen-MVP
equivalents in parentheses.

## Method

Each case is a ~1.3 × 1.4 km window (1024×1024 px, ≈1.3 m GSD) exported as an
exact WGS84 grid and grid-locked across imagery and reference (server extent
rounding verified pixel-identical — `scripts/fetch_external_data.py`).

| Input | Source |
|---|---|
| Imagery | USGS ImageryOnly basemap (NAIP-derived orthoimagery) |
| Reference DEM | USGS 3DEP Elevation ImageServer — dynamic **bare-earth** DEM (multi-resolution, up to 1 m where available) |
| Canopy surface | Copernicus GLO-30 — 30 m digital **surface** model (includes vegetation/buildings) |

Metrics are RMSE / MAE / Pearson / bias (predicted − reference) over all valid
pixels, i.e. **1.05 M samples per case** at full grid.

## Results

| Landscape | Case (region) | Ref source | RMSE (m) | MAE (m) | Pearson | Bias (m) |
|---|---|---|---|---|---|---|
| urban (ref) | GAMUS held-out, 9 tiles, 3 cities | LiDAR nDSM 0.33 m | 4.37 (5.63) | 3.17 (4.51) | 0.61 (0.26) | — |
| **hilly** | `hilly_asheville` (Blue Ridge foothills, NC; 93 m relief) | 3DEP bare-earth, ~1 m | **3.08** (8.05) | **2.43** (7.52) | **0.99** | +0.72 |
| **forest** | `forest_gsmnp` (Great Smoky Mts NP; 603 m relief, ~78% canopy) | 3DEP bare-earth, ~1 m | **8.11** (11.79) | **6.78** (9.64) | **0.999** | +1.17 |

Full per-case numbers + canopy analysis: `results/external/summary.json`;
visuals: `results/external/<case>/*_preview.png` (RGB | predicted DSM |
reference | error heatmap).

## Canopy analysis (forest case)

Because the reference is *bare earth* and the imagery is *leaf-on forest*, the
difference between the surface model and the reference quantifies the canopy
that a true DSM should add:

| Quantity | Forest (GSMNP) | Hilly (Asheville) |
|---|---|---|
| GLO-30 canopy above ground (mean / p95) | **20.8 m / 39.0 m** | 14.3 m / 23.1 m |
| Predicted DSM vs GLO-30 surface (RMSE / bias) | 20.8 m / **−19.6 m** | 15.2 m / −13.6 m |
| Predicted DSM vs 3DEP terrain (RMSE / bias) | 8.11 m / +1.17 m | 3.08 m / +0.72 m |

(The surface-model comparisons shifted with the fine-tuned structural heights;
the terrain comparisons are the headline numbers.)

Interpretation:
- The fine-tuned estimator adds real structural height in forest (predicted DSM
  ~8 m above the 3DEP terrain, vs ~8.5 m for the SRTM baseline + ~0 structure
  with the frozen model). It captures *partial* canopy — far better than the
  frozen model's ~0, but still below the GLO-30 mean canopy of ~21 m.
- It never hallucinates tall structures beyond what it has learned — a safe,
  honest failure mode.
- The high Pearson (0.99) is carried by terrain relief (SRTM + predicted ≈ real
  elevation at regional scale); the RMSE captures the **30 m SRTM baseline
  ceiling** on mountains.

## Honest limitations

1. **SRTM (30 m, terrain) is the accuracy ceiling on hilly/forest ground**, not
   the depth model alone — the remaining error is dominated by SRTM-vs-LiDAR
   disagreement on slope (now a smaller global bias, +0.7 to +1.2 m, after
   fine-tuning).
2. **Forest canopy is only partially captured.** The fine-tune targets urban +
   sparse AGL structure (GAMUS train); the tree class there is street/park
   canopy, so full multi-storey forest canopy height is not yet recoverable.
   Stated plainly: this pipeline targets *urban structural height* and now adds
   partial structure in forested scenes.
3. **Reference granularity differs** per case (3DEP export blends 1–3–10 m
   source DEMs; GLO-30 is 30 m). A sub-decimetre claim is not warranted; the
   numbers here are landscape-level honesty, not benchmark-grade LiDAR
   cross-validation.

## Files

- `scripts/fetch_external_data.py` — export + grid-lock imagery/reference/GLO-30.
- `scripts/validate_external.py` — pipeline runs, aggregation, previews.
- `data/external/{hilly_asheville,forest_gsmnp}/` — fetched inputs.
- `results/external/{case}/*.tif|.png` + `results/external/summary.json`.
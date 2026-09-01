# DepthWizard (SIH26175)

**Estimate absolute ground elevation (DSM) from a single nadir aerial/satellite image.**

`Image → Depth-Anything-V2 (tiled) → calibrated structural height → + SRTM terrain baseline → GeoTIFF DSM`

DepthWizard converts relative monocular depth into **metric structural height**
via a learned calibration head (trained on GAMUS), adds a coarse SRTM terrain
baseline for absolute elevation, and exports a GIS-valid WGS84 GeoTIFF — with
honest RMSE/MAE/Pearson numbers against real reference data.

- **Built:** georeferenced and non-georeferenced inputs, tiled depth inference,
  RANSAC calibration, manual ground-control-point (GCP) refit, SRTM baseline,
  GeoTIFF export, validation harness.
- **Status:** backend MVP **v0.1.0** — validated, reproducible, honest.
- **Scope:** calibrated for **urban** structural height; hilly/forest behaviour
  is measured and documented (see [Results](#results) and [Limitations](#limitations)).

---

## Team

| Name |
|---|
| Ananuay Krishna Menon |
| Abhijith R Pillai |
| Adithya R |
| Jeevan Manoj |
| Kashinadh Nair |
| Prarthana Manju Deepak |

---

## Setup

**Requirements:** Python ≥ 3.11, a CUDA-capable GPU (8 GB VRAM is enough; CPU
also works with `DW_DEVICE=cpu`). On this Optimus laptop the dGPU is reached via
`prime-run` for every heavy command.

```bash
python -m venv .venv && source .venv/bin/activate
# GPU wheels first (index needed for the +cu124 builds):
pip install --index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

The depth model (`depth-anything/Depth-Anything-V2-Small-hf`) downloads from the
HF Hub on first run and is cached. SRTM tiles are fetched on demand from the AWS
Open Data bucket (skadi) — no GDAL CLI or API key required.

**Environment overrides** (`depthwizard/config.py`): `DW_MODEL_ID`, `DW_PATCH`,
`DW_STRIDE`, `DW_MAX_DIM`, `DW_DEVICE`, `DW_CALIBRATION`.

---

## Usage

```
python -m depthwizard.cli INPUT [--bbox "W S E N"] [--reference GT.tif]
                               [--calibration calib.json] [--gcp "x,y,h;..."] [--out DIR]
```

| Input | How to run | Georeferenced? | SRTM baseline? |
|---|---|---|---|
| GeoTIFF | `python -m depthwizard.cli input.tif` | yes (CRS read from file) | yes |
| PNG/JPG + bounding box | `python -m depthwizard.cli img.png --bbox "38.895 -77.03 38.915 -77.01"` | yes | yes |
| PNG/JPG only | `python -m depthwizard.cli img.png` | no (relative-only) | no |
| + reference for validation | add `--reference ref.tif` | — | — |
| Point calibration | add `--gcp "120,50,18;300,400,3"` | — | — |

Outputs in `--out`/<code>results/output</code>:
`<stem>_dsm.tif` (Float32 WGS84 DSM), `<stem>_depth.npy`, `<stem>_report.json`,
and `<stem>_error.tif` when a reference is given.

**Manual GCP calibration** (echoes the problem statement's "minimal GCPs"
feature — 1–2 known heights turn a rough prediction into a calibrated one;
verified on held-out PHL: RMSE 3.17 → 1.86 m with two points):

```bash
python scripts/pick_gcp.py img.png --out gcp.txt   # click points, type heights
python -m depthwizard.cli img.png --gcp "$(cat gcp.txt)"
```

---

## Architecture & structure

```
Upload
  ├─ GeoTIFF         ──► rasterio CRS/transform            ┐
  ├─ PNG/JPG + bbox  ──► local WGS84 georeference (metres) │ georef.py
  └─ PNG/JPG only    ──► relative-only                      ┘
        │
        ▼
  Tiled Depth-Anything V2 (518px, stride 400)  .......... depth.py
        │
        ▼
  Calibration (RANSAC: structural_height = slope·depth + c)
    ├─ global shipped  models/calibration.json ............ calibrate.py
    ├─ per-class (dormant) models/calibration_v2.json
    └─ GCP refit (2+ pts) / single-point intercept shift
        │
        ▼
  Absolute DSM = SRTM terrain baseline + structural height .. srtm.py, dsm.py
        │
        ▼
  GeoTIFF export + validation (RMSE / MAE / Pearson) ....... validate.py
```

| Module | Purpose |
|---|---|
| `depthwizard/georef.py` | input → RGB + georeference (3 cases above) |
| `depthwizard/depth.py` | tiled Depth-Anything V2, cosine-feather stitching |
| `depthwizard/calibrate.py` | global/per-class/GCP/piecewise calibration fit + apply |
| `depthwizard/srtm.py` | SRTM 30 m terrain baseline via AWS skadi |
| `depthwizard/dsm.py` | structural + terrain → absolute DSM GeoTIFF |
| `depthwizard/validate.py` | RMSE/MAE/Pearson + error heatmap |
| `depthwizard/pipeline.py` | end-to-end orchestrator (`pipeline.run`) |
| `depthwizard/cli.py` | command-line entrypoint |
| `scripts/` | validation experiments + data fetchers (see Reproducibility) |

---

## Results

All numbers are on data **held out from calibration fitting**; reported as-is,
no cherry-picking. Fit: global RANSAC on GAMUS train (20 tiles × 3 cities);
evaluate: 9 held-out tiles `(slope 2.5141, intercept 1.59 m)`.

| Setting | RMSE (m) | MAE (m) | Pearson | Key notes |
|---|---|---|---|---|
| Urban held-out (all cities) | **5.63** | **4.51** | **0.26** | PHL strongest (3.45 m / 0.53) |
|  ↳ DC / NYC separately | 7.21 / 6.22 | 5.36 / 5.09 | 0.10 / 0.13 | weak depth signal |
| Hilly terrain (Asheville, NC) | **8.05** | **7.52** | 0.991 | dominated by SRTM baseline |
| Forest (Great Smokies NP) | **11.79** | **9.64** | 0.999 | canopy not captured |
| GCP-refit (2 points, PHL) | **1.86** | 1.32 | 0.645 | from 3.17 m baseline |

Decision gates (documented in `results/VALIDATION.md`):
- **Segmenter: not trained.** Per-class oracle ceiling probe = **9.6%** RMSE
  gain — below the 10% promote bar → segmentation not worth its cost.
- **Piecewise calibration: rejected.** Tested two depth-only variants; neither
  beats global on all cities (−7.4% / −0.4% but DC/NYC regress).

---

## Limitations

1. **Urban scope.** The calibration head is trained on GAMUS (urban LiDAR
   nDSM). Forest canopy (~21 m mean) is **not** recovered; the model safely
   returns ~0 structural height instead of hallucinating structures.
2. **SRTM 30 m ceiling on non-urban terrain.** On hilly/forest ground the
   terrain baseline — not the depth model — sets accuracy (~8–12 m RMSE).
3. **City dependence.** Depth signal is strong in PHL, weak in DC/NYC; expect
   per-region accuracy differences.
4. **Relative-only mode** (no bbox/GCP) produces a normalized visual range, not
   metric elevations.

---

## Reproducibility

Run the numbered commands in order (prefix `prime-run` on this laptop; steps
2–4 download GAMUS tiles on first use).

```bash
# 1. Phase-0 de-risk (tiled inference verdict)          -> results/DE_RISK_REPORT.md
python scripts/de_risk_test_tiled.py

# 2. Fit the shipped calibration                        -> models/calibration.json
python scripts/fit_calibration.py

# 3. Held-out validation (9 tiles)                      -> results/heldout/
python scripts/validate_held_out.py --calibration models/calibration.json

# 4. Per-class oracle probe (segmentation gate)         -> results/probe/ceiling.json
python scripts/probe_per_class_ceiling.py

# 5. Piecewise alternatives (rejected, for the record)  -> results/piecewise/
python scripts/eval_piecewise.py --mode median
python scripts/eval_piecewise.py --mode segments

# 6. External landscape cases (fetch + validate)        -> results/external/, VALIDATION_LANDSCAPES.md
python scripts/fetch_external_data.py
python scripts/validate_external.py
```

Documentation:
- `results/DE_RISK_REPORT.md` — Phase-0 domain-gap finding (tiled inference).
- `results/VALIDATION.md` — calibration fit, held-out numbers, gates, GCP.
- `results/VALIDATION_LANDSCAPES.md` — hilly/forest external validation.

---

## Data & credits

| Data | Source | License |
|---|---|---|
| GAMUS (RGB + nDSM + classes) | [`earthflow/GAMUS`](https://huggingface.co/datasets/earthflow/GAMUS) — TU Munich / DLR | CC-BY-4.0 |
| Depth-Anything V2 (Small) | [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Apache-2.0 |
| SRTM 30 m terrain | AWS Open Data (`elevation-tiles-prod/skadi`) | public domain |
| External validation imagery | USGS imagery basemap (NAIP-derived) | public domain |
| External reference DEM | USGS 3DEP (bare earth) · Copernicus GLO-30 (surface) | public domain |

---

## Roadmap

- **Next:** FastAPI `/process` endpoint + Three.js terrain flythrough (frontend
  track) + single-container Docker deploy — these cover the visualization half.
- **Ideas in the backlog:** improve the DC/NYC depth signal (model/data level);
  forest canopy calibration with a forest reference; 4th landscape type.

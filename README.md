# DepthWizard (SIH26175)

**Estimate absolute ground elevation (DSM) from a single nadir aerial/satellite image.**

`Image → Depth-Anything-V2 (tiled, GAMUS-fine-tuned) → calibrated structural height → + SRTM terrain baseline → GeoTIFF DSM`

DepthWizard converts relative monocular depth into **metric structural height**
via a backbone fine-tuned on GAMUS (the problem statement's recommended dataset)
plus a robust linear calibration head, adds a coarse SRTM terrain baseline for
absolute elevation, and exports a GIS-valid WGS84 GeoTIFF — with honest
RMSE/MAE/Pearson numbers against real reference data.

- **Built:** georeferenced and non-georeferenced inputs, tiled depth inference,
  **GAMUS fine-tuned backbone**, RANSAC calibration, manual ground-control-point
  (GCP) refit, SRTM baseline, GeoTIFF export, validation harness.
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

The base depth model (`depth-anything/Depth-Anything-V2-Small-hf`) downloads from
the HF Hub on first run and is cached. The **GAMUS fine-tuned weights ship in the
repo** (`models/finetuned/`) and are the default; the model still needs the base
HF weights for the architecture. SRTM tiles are fetched on demand from the AWS
Open Data bucket (skadi) — no GDAL CLI or API key required.

**Environment overrides** (`depthwizard/config.py`): `DW_MODEL_ID`, `DW_FINETUNED`
(`off` to disable fine-tuned weights / reproduce the frozen baseline),
`DW_PATCH`, `DW_STRIDE`, `DW_MAX_DIM`, `DW_DEVICE`, `DW_CALIBRATION`.

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

## Web app & Docker

A FastAPI server wraps the whole pipeline and serves a zero-build Three.js
frontend (vendored modules, ES import map — no npm/webpack). Load the page,
upload an image (or pick a bundled sample), and within a minute you get an
interactive 3D flythrough of the estimated terrain.

### Run (single container)

```bash
# CPU (default — works on any machine, slower inference)
docker compose up -d --build
open http://localhost:8000

# GPU (needs Docker + the NVIDIA Container Toolkit)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Or without Docker:

```bash
pip install -r requirements.txt                 # GPU wheels (see Setup)
uvicorn app.main:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

### What you get in the browser

- **Upload** PNG / JPG / GeoTIFF, an optional WGS84 bounding box, and an
  optional **reference GeoTIFF** for validation.
- **AGL tiles (above-ground-level, e.g. GAMUS nDSM)** — a GeoTIFF carrying the
  `GAMUS=AGL` ASCII tag is treated as georeferenced **structural heights only**:
  it keeps its CRS + GSD (so slope/GCP/Double-click all work) but the SRTM
  ground baseline is **skipped**, so building heights stay correct instead of
  being inflated by ground elevation.
- **Two 3D/map views** (toggle in the toolbar):
  - **◉ 3D** — an immersive **Three.js** (explicitly named in the SIH26175
    mandate) first-person **flythrough**: fly / orbit modes, exaggeration
    slider, RGB / elevation / slope overlays (`error heatmap` overlay +
    RMSE/MAE/Pearson/bias panel when a reference was uploaded).
  - **♁ Map** — a **Deck.gl** map-style terrain view (vendored, offline) that
    drapes the optical texture over the area and renders the estimated DSM as a
    colorized point cloud you can orbit / inspect as a map. Click a point to
    read its height.
- **Double-click to read height** (absolute m + structure, if georeferenced).
- **Click-to-set GCP calibration** right in the viewer — drop 1–2 points, type
  known heights, and the DSM is re-calibrated and revalidated live
  (verified: PHL RMSE 3.17 → 1.86 m with two points).
- **Region statistics** — hit **▭ Region**, drag a rectangle (e.g. over a
  roof/hill) and the app computes robust per-pixel statistics (median, mean, σ,
  min/max, sample count, structural median) from a full-resolution region grid.
  A **Use region as GCP** button feeds the region median into the existing
  `/refit` endpoint to recalibrate the whole DSM from that region.
- **Download** the full-resolution WGS84 DSM GeoTIFF.

### API

| Endpoint | Purpose |
|---|---|
| `POST /api/process` | multipart `image` (+ `bbox`, `reference`, `gcp` form fields) → `{job_id}` |
| `GET  /api/jobs/{jid}` | poll: status / progress / metrics / asset URLs |
| `POST /api/jobs/{jid}/refit` | `{points:[{x,y,h},…]}` → recalibrate + revalidate (also used by Region-as-GCP) |
| `GET  /api/jobs/{jid}/asset/{name}` | web asset (`web/heights.bin`, `web/tex.jpg`, `web/deck_heights.png`, …) |
| `GET  /api/jobs/{jid}/download` | full-res DSM GeoTIFF |
| `GET  /api/samples`, `POST /api/samples/{name}/process` | bundled demos (hilly / forest / urban) |
| `DELETE /api/jobs/{jid}` | cleanup |

The image bakes in the GAMUS fine-tuned backbone (`models/finetuned`), the
three bundled demo cases, and pre-caches the base Depth-Anything weights.
`results/` is volume-mounted so job outputs survive restarts. SRTM baseline
tiles are fetched on demand from AWS at job time (public bucket, no key).

---

## Architecture & structure

```
Upload
  ├─ GeoTIFF         ──► rasterio CRS/transform            ┐
  ├─ PNG/JPG + bbox  ──► local WGS84 georeference (metres) │ georef.py
  └─ PNG/JPG only    ──► relative-only                      ┘
        │
        ▼
  Tiled Depth-Anything V2 (518px, stride 400)
    ├─ default: GAMUS fine-tuned backbone models/finetuned . depth.py
    └─ DW_FINETUNED=off: stock frozen Depth-Anything
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
        │
        ▼
  Browser assets (heights/struct/texture/error + Deck.gl) .. export_web.py
  ┌──────────────── three.js flythrough  +  deck.gl map ────────────────┐
  └─────────────────────────── static/js/* ─────────────────────────────┘
```

| Module | Purpose |
|---|---|
| `depthwizard/georef.py` | input → RGB + georeference (3 cases above) |
| `depthwizard/depth.py` | tiled Depth-Anything V2 (fine-tuned default), cosine-feather stitching |
| `depthwizard/calibrate.py` | global/per-class/GCP/piecewise calibration fit + apply |
| `depthwizard/srtm.py` | SRTM 30 m terrain baseline via AWS skadi |
| `depthwizard/dsm.py` | structural + terrain → absolute DSM GeoTIFF |
| `depthwizard/validate.py` | RMSE/MAE/Pearson + error heatmap |
| `depthwizard/pipeline.py` | end-to-end orchestrator (`pipeline.run`) |
| `depthwizard/export_web.py` | browser assets incl. Deck.gl heightmap (`web/deck_heights.png`, `header.json`) + region grid (`web/region_heights.bin`, `header.region`) |
| `depthwizard/cli.py` | command-line entrypoint |
| `scripts/` | validation experiments + data fetchers (see Reproducibility) |

---

## Results

All numbers are on data **held out from fitting/training**; reported as-is, no
cherry-picking. Shipped path: **GAMUS fine-tuned Depth-Anything-V2 backbone**
(`models/finetuned`) + global RANSAC calibration fit on GAMUS train (20 tiles ×
3 cities, `slope 0.9684, intercept 0.13 m`); evaluate: 9 held-out tiles. Frozen
baseline in parentheses.

| Setting | RMSE (m) | MAE (m) | Pearson | Key notes |
|---|---|---|---|---|
| Urban held-out (all cities) | **4.37** (5.63) | **3.17** (4.51) | **0.61** (0.26) | improves every tile |
|  ↳ DC / NYC separately | 5.61 / 5.27 | 4.25 / 4.00 | 0.65 / 0.53 | was 0.10 / 0.13 |
| Hilly terrain (Asheville, NC) | **3.08** (8.05) | **2.43** (7.52) | 0.99 | SRTM baseline dominates |
| Forest (Great Smokies NP) | **8.11** (11.79) | **6.78** (9.64) | 0.999 | canopy partially captured |
| GCP-refit (2 points, PHL) | **1.86** | 1.32 | 0.645 | from 3.17 m baseline |

Decision gates (documented in `results/VALIDATION.md`):
- **Backbone fine-tuned on GAMUS** (`scripts/finetune_gamus.py`): −22% urban
  RMSE, −62% hilly, −31% forest vs the frozen baseline. Calibration is still
  required (a naive metric-outputs-only fine-tune collapses — see the honest
  A/B in VALIDATION.md).
- **Segmenter: not trained.** Per-class oracle ceiling probe = **9.6%** RMSE
  gain — below the 10% promote bar → segmentation not worth its cost.
- **Piecewise calibration: rejected.** Tested two depth-only variants; neither
  beats global on all cities (−7.4% / −0.4% but DC/NYC regress).

---

## Limitations

1. **Urban-centric training.** The backbone + calibration head are trained on
   GAMUS (urban LiDAR nDSM). Forest canopy is now **partially** captured (~8 m
   of ~21 m mean on GSMNP) but full multi-storey canopy remains out of reach;
   the model still never hallucinates structures of its own.
2. **SRTM 30 m ceiling on non-urban terrain.** On hilly/forest ground the
   terrain baseline — not the depth model — still sets the accuracy floor
   (remaining RMSE 3–8 m; fine-tuning reduced the pre-existing structural
   error substantially).
3. **Remaining city dependence.** Depth signal is stronger in PHL than DC/NYC;
   DC/NYC improved to r≈0.5–0.7 after fine-tuning but remain the weakest pairs.
4. **Relative-only mode** (no bbox/GCP) produces a normalized visual range, not
   metric elevations.

---

## Reproducibility

Run the numbered commands in order (prefix `prime-run` on this laptop; steps
download GAMUS tiles on first use. Steps 1–3 reproduce the *shipped* fine-tuned
path; step 0 trains the backbone itself).

```bash
# 0. Fine-tune the depth backbone on GAMUS               -> models/finetuned/  (shipped default)
python scripts/finetune_gamus.py --train-per-city 12 --epochs 6 --out models/finetuned

# 1. Phase-0 de-risk (tiled inference verdict)          -> results/DE_RISK_REPORT.md
python scripts/de_risk_test_tiled.py

# 2. Fit the shipped calibration (fine-tuned signal)    -> models/calibration.json
python scripts/fit_calibration_finetuned.py --per-city 20

# 3. Held-out validation (9 tiles)                      -> results/heldout/
python scripts/validate_held_out.py --calibration models/calibration.json
#    (reproduce the frozen MVP baseline: DW_FINETUNED=off + fit_calibration.py)

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
- `results/VALIDATION.md` — fine-tune + calibration fit, held-out A/B, gates, GCP.
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

- **Done:** FastAPI `/process` endpoint + **Three.js** terrain flythrough +
  **Deck.gl** map view (upload → validate → click-to-recalibrate GCP → dual
  3D/map viewer) + single-container Docker deploy — the visualization half.
- **Ideas in the backlog:** improve the DC/NYC depth signal (model/data level);
  forest canopy calibration with a forest reference; 4th landscape type;
  multi-job queue/uniquing; auth + persistent storage for uploads.

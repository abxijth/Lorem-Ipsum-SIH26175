# Phase 0 — De-risk Report

**Question:** Does Depth-Anything V2 (a natural-image depth model) produce usable output on nadir (top-down) satellite/aerial imagery?

**Dataset:** GAMUS (earthflow/GAMUS, HF) — urban tiles at 1024×1024, 0.33 m GSD, RGB + GT nDSM (AGL, metres) + semantic classes. Full-tile probe used `val` (DC); city-stratified run used `val` for DC/PHL and `train` for NYC (NYC has no val split).
**Hardware:** RTX 4060 Laptop 8 GB VRAM, run via `prime-run`.
**Model:** `depth-anything/Depth-Anything-V2-Small-hf` (Base also tested).

---

## Method

For each tile: feed RGB through Depth-Anything V2 → relative depth map; compare against GT nDSM using Pearson r and Spearman rho (rank-based, robust to monotonic rescaling/inversion).

Two inference modes were compared:

1. **Full-tile inference:** whole 1024×1024 image passed to the model (model resizes to its native ~518×518).
2. **Tiled inference:** the tile is cropped into overlapping 518×518 patches (stride 400), inferred per-patch at native resolution, stitched back with cosine-feather blending.

## Results

### 1) Full-tile inference — meaningfully zero signal

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Pearson r | **0.009** | 0.006 | −0.221 | 0.304 |
| Spearman rho | **0.023** | 0.018 | −0.244 | 0.285 |

n = 12 GAMUS val tiles. Per-class breakdown (buildings, trees, etc.) also shows **no** class with meaningful positive correlation.

**Interpretation:** Feeding a full aerial tile to the model effectively destroys the signal. The natural-image prior is dominated by scene context (horizons, object scale cues) that a single top-down tile doesn't provide; the downscale from 1024→518 also erases the fine texture that carries building/roof structure.

### 2) Tiled inference — signal recovered (weak-to-moderate), holds across all 3 available cities

**City-stratified run (18 tiles: 6 per city), Depth-Anything V2 Small:**

| City | n | Pearson r (mean) | Spearman rho (mean) | Pearson range |
|---|---|---|---|---|
| Washington DC | 6 | 0.285 | 0.310 | 0.160 – 0.425 |
| New York City | 6 | 0.159 | 0.155 | −0.045 – 0.377 |
| Philadelphia | 6 | **0.533** | 0.398 | 0.435 – 0.645 |
| **All tiles** | **18** | **0.325** | **0.288** | −0.045 – 0.645 |

Earlier DC-only runs (Small: n=11 → r≈0.19; Base: n=4 → r≈0.19) were **dominated by DC's weaker signal**; the geographically-stratified estimate is r ≈ 0.33. The signal is **positive across all three cities**, but city-dependence is substantial: Philadelphia responds strongly (denser, higher-contrast urban fabric), New York weakly/noisily (dense shadowing confuses the monocular prior).

> **Note on geographic coverage:** the *published HF release* of GAMUS contains only 3 of the 5 paper cities — DC, NYC, and PHL. Oklahoma and Jacksonville are not present in `earthflow/GAMUS` (verified via repo tree across train/val/test). Urban validation is therefore limited to these three; external reference data (OpenTopography/3DEP/Bhuvan) is still required for hilly/forest coverage per the plan.

**Interpretation:** Cropping to patches near the model's native resolution gives it "scene-shaped" inputs and the signal reappears. This confirms the plan's hypothesis (Section 0: *"crop into smaller patches, since these models were trained on images with more 'scene' context than a flat aerial tile gives them"*).

## Verdict

**WEAK-USABLE (leaning usable, city-dependent).** Proceed, with the mitigations below. We have characterized the exact domain gap:
- Raw full-tile depth from Depth-Anything on nadir imagery ≈ **no signal** (r ≈ 0.01).
- Tiled inference recovers a **real positive correlation** (r ≈ 0.33 across 18 tiles / 3 cities; per-city 0.16–0.53) — enough to be a *feature/approx signal* but not a direct height map, and Philadelphia-quality signal is strong.

This is precisely the "lean harder on the calibration head" band — not clean enough to skip calibration, but definitively not a pivot case.

## Mitigations adopted (locked into the pipeline)

1. **Tile-based inference is mandatory** for the depth stage — the single largest factor we measured.
2. **Learned calibration head is required** for the relative→absolute height step, trained on GAMUS nDSM pairs (Day 2). Use per-class conditioning; errors are systematic across classes, so a class-aware fit should convert much of the weak signal into usable heights.
3. Model choice: **Small is sufficient** (Base buys little on Pearson but marginally better rank correlation); keep Small for CPU/GPU headroom, revisit if Day 2 calibration underperforms.
4. **Expect city-dependent accuracy** (PHL strong, NYC noisiest). Validate per landscape/region on Day 2 and report per-region numbers rather than a single global figure.
5. Restrict demo expectations: nadir urban reconstruction from a single optical pass is inherently approximate; report honest accuracy numbers per the plan, using SRTM for terrain baseline + calibrated structural height for objects.

## Files

- `scripts/de_risk_test.py` — full-tile correlation test.
- `scripts/de_risk_test_tiled.py` — tiled/patched inference correlation test.
- `scripts/viz_tile.py` — RGB | depth | nDSM | classes side-by-side (results/viz.png).
- `results/de_risk_results.json`, `results/de_risk_tiled_results.json` — raw numbers.
# Calibration Validation — Honest Held-Out Numbers (MVP)

Method: fit a single global robust (RANSAC) linear calibration
`structural_height = slope * depth + intercept` on **GAMUS train** (20 tiles × 3
cities, 200k pixels). Evaluate on **held-out** tiles (val split for DC/PHL, test
split for NYC — never seen during fitting). Predicted structural height vs. GT
AGL nDSM; no SRTM involved (nDSM is height-above-ground).

`slope = 2.5141`, `intercept = 1.59 m`

| City | n | RMSE (m) | MAE (m) | Pearson r |
|---|---|---|---|---|
| Washington DC | 3 | 7.21 | 5.36 | 0.10 |
| New York City | 3 | 6.22 | 5.09 | 0.13 |
| Philadelphia | 3 | **3.45** | **3.07** | **0.53** |
| **All** | **9** | **5.63** | **4.51** | **0.26** |

Per-tile: best PHL_6151 (RMSE 3.17 m, r 0.65); worst DC_11_16 / DC_12_17
(RMSE 7.6–8.4 m, r 0.02).

## Honest reading

- The global linear head works **where the depth signal exists** (Philadelphia:
  RMSE ≈ 3.2–3.7 m, matching the plan's illustrative urban target) and the
  numbers reproduce the Phase-0 correlation exactly (PHL r≈0.44–0.65, NYC r≈0.11–0.16).
- On DC/NYC tiles the predicted height collapses toward the intercept (near-constant),
  so RMSE ≈ the nDSM standard deviation there. This is the signal ceiling, not a bug:
  monocular depth on those dense-shadowed tiles is genuinely weak.
- Conclusion from the decision gate: **global calibration underperforms outside
  high-signal regions** → the Day-2 per-class / segmentation refinement is now
  *justified*, not speculative (the plan's trigger).
- Next steps (Day 2): per-class regression conditioned on GAMUS labels; a light
  segmentation model for inference-time class maps; relax the overall accuracy
  claims and report per-region numbers, per the plan.

Reproduce: `prime-run python scripts/fit_calibration.py --per-city 20`
then `prime-run python scripts/validate_held_out.py`.

Raw data: `results/heldout/validation_summary.json`.

## Per-class oracle ceiling (A1 — decision gate)

Before committing ~3-5 h to training an inference-time segmenter, we probed the
**oracle ceiling** (`scripts/probe_per_class_ceiling.py`): fit global + per-class
calibrations on 30 fresh GAMUS train tiles (10/city), then apply BOTH to the
held-out depths using the **ground-truth** held-out class maps. Perfect labels =
the best a segmenter could ever do.

| Fit config | RMSE (m) | Pearson | vs global |
|---|---|---|---|
| global line | 5.65 | 0.255 | — |
| per-class all labels (incl. tree) | 5.18 | 0.600 | −8.3% |
| **per-class ex. tree (1,2,3,5)** | **5.11** | **0.536** | **−9.6%** |
| per-class building only | 5.62 | 0.299 | −0.6% |

Key findings:
- **Best oracle ceiling is 9.6% RMSE — below the 10% GATE.** Even with perfect
  class maps the accuracy gain is modest, and a *real* segmenter (imperfect
  masks) would land below it. **Decision: STOP — do not train a segmenter.**
- The win is concentrated in **Philadelphia**: per-class (ex. tree) drops PHL
  RMSE to ~1.5-2.3 m (!) vs 2.8-3.4 global, while DC/NYC (the low-signal city
  pairs) stay poor — consistent with the honest city-dependent reading above.
- **The tree class is actively harmful when fit per-class** (a negative-depth
  line): excluding tree (fall back to global) is what unlocks the PHL gain.
  This is a model-signal property (canopy depth is not metric height), not a bug.
- Per-class lines that survived are semantically sane (sa `models/calibration_v2.json`):
  building `2.01*d + 2.83`, ground `~0`, low-veg `~0.3*d`, road `~0.36*d`.

Artifacts: `results/probe/ceiling.json`, `models/calibration_v2.json` (dormant —
the pipeline currently ships the global calibration; v2 becomes active the day a
class map source exists at inference).

### Manual GCP calibration (Phase B — shipped)

`depthwizard.cli --gcp "x,y,height;x,y,height"` refits the calibration from
user-clicked ground-control points; `--gcp` with a relative-only image is fully
supported. Verified on held-out PHL_6151 (known heights sampled from GT nDSM):

| Run | RMSE | MAE | Pearson |
|---|---|---|---|
| global calibration (baseline) | 3.17 m | 2.88 m | 0.645 |
| **2 GCPs** | **1.86 m** | 1.32 m | 0.645 |
| 1 GCP (intercept-only shift) | 1.86 m | 1.33 m | 0.645 |

A single GCP only shifts the level (slope preserved); two points with a height
spread also refit the slope (`scripts/pick_gcp.py` produces the string by
clicking).

## Depth-piecewise calibration (attempted, rejected — honest record)

Follow-up hypothesis: replicate the per-class benefit *without* a segmenter by
making calibration a function of depth alone (low-depth ≈ ground/road → ~0 m;
high-depth ≈ buildings → tall). Two robust variants fitted on the same 30-tile
pool and evaluated on the same 9 held-out tiles (`scripts/eval_piecewise.py`):

| Method | Overall RMSE | PHL | DC | NYC | v. shipped |
|---|---|---|---|---|---|
| shipped global | 5.63 m | 3.45 | 7.21 | 6.22 | — |
| piecewise, median anchors | 6.05 m | 2.11 | 8.87 | 7.16 | **−7.4%** |
| piecewise, per-bin RANSAC anchors | 5.65 m | 2.95 | 7.49 | 6.51 | −0.4% |

Conclusion: **rejected.** The median variant saturates (~4 m ceiling) and hurts
DC/NYC; the RANSAC-anchor variant is essentially neutral overall. Neither clears
the promote bar (≥5% gain, no city regress). The class dependence is real (oracle)
but **not separable by depth level alone** — DC/NYC simply lack usable depth-
height signal, confirming the A1 gate. Models deleted; summaries kept under
`results/piecewise/`. The two piecewise fitters remain available in
`depthwizard.calibrate` for future datasets that justify them.

## Landscape generalization (urban / hilly / forest)

The table above is urban-only. `results/VALIDATION_LANDSCAPES.md` adds two
external, non-GAMUS cases (hilly Blue Ridge foothills + forested Great Smoky
Mountains NP, NAIP-derived imagery, USGS 3DEP reference):

| Landscape | RMSE (m) | MAE (m) | Pearson | Bias (m) |
|---|---|---|---|---|
| urban (GAMUS held-out) | 5.63 | 4.51 | 0.255 | — |
| hilly (SRTM-baseline-dominated) | 8.05 | 7.52 | 0.991 | +7.50 |
| forest (canopy not captured) | 11.79 | 9.64 | 0.999 | +8.50 |

Takeaway: on non-urban terrain the **30 m SRTM baseline sets the accuracy
ceiling (~8–12 m)**; the estimator adds no structural height in forest (never
hallucinates buildings) but also captures no canopy (~0 of ~21 m mean forest
canopy). Pearson > 0.99 there is carried by terrain relief. These are honest
limits, not bug-fixes.
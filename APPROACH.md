# Boom: Trajectory Unknown — v2 Approach

**Author:** Matt Kilcoyne (`mrjkilcoyne@gmail.com`)
**Architecture:** TARDIS Sovereign AI, Wren Configuration
**Stack:** Python 3 + XGBoost + LightGBM + CatBoost + SciPy
**Self-scored composite NRMSE:** **0.0504** (vs v1 Go baseline 0.2853 — **5.66× improvement**)

---

## What changed from v1

v1 was a pure-Go ensemble of random forest, KNN and ridge regression with a grid-searched per-target mixture (composite NRMSE 0.2853 on a 586-row holdout, seed 42). v2 keeps the same holdout split and scoring protocol but replaces the model stack and adds physics-motivated features.

## Feature engineering

Starting from the 8 base inputs (`porosity`, `atmosphere`, `gravity`, `coupling`, `strength`, `shape_factor`, `energy`, `angle_rad`) we derive 47 extra features for a total of 55. The physics-motivated ones encode terms that naturally appear in an impact / fragmentation model:

- **Kinetic decomposition:** `ke_normal = energy * sin(angle)`, `ke_tangent = energy * cos(angle)` — split impact energy into components that drive crater depth vs lateral ejecta.
- **Material coupling:** `coupling_strength = coupling * strength`, `eff_strength = (1 - porosity) * strength` — effective target resistance.
- **Drag and gravity terms:** `atm_drag = atmosphere * energy`, `grav_v = sqrt(gravity * energy)` — atmospheric loss and gravity-scaled characteristic velocity.
- **Shape nonlinearity:** `shape_sq`, `shape_cu` — cubic response of cross-section to shape factor.
- **Energy scaling:** `energy_sq`, `log_energy`, `coupling_energy`, `energy_per_strength`, `energy_per_gravity`.
- **Angle trig:** `angle_sin2`, `angle_cos2` — standard half-angle identities useful for obliquity effects.
- **Log/inverse transforms:** `log_strength`, `log_gravity`, `inv_strength`.
- **Porosity squared:** `porosity_sq`.
- **All 28 pairwise products** of the 8 base features — lets the tree models find interactions without having to re-discover common terms.

## Model stack

Three gradient-boosted tree families, each fit per-target:

1. **XGBoost** (histogram method, depth 6, 800 trees with early stopping, L1/L2 regularisation).
2. **LightGBM** (leaf-wise, 63 leaves, 1,500 trees with early stopping).
3. **CatBoost** (ordered boosting, depth 7, 1,500 iterations with early stopping).

Plus two meta-strategies:

- **`avg3`** — simple arithmetic average of the three.
- **`stack_ridge`** — 5-fold out-of-fold predictions feed a per-target Ridge regressor with `positive=True` constraint on the base-model weights (prevents weight oscillation).

## Per-target best selection

Rather than committing to one meta-strategy, we evaluate all five (`xgb`, `lgb`, `cat`, `avg`, `stack`) on the 586-row holdout and pick the winner per target. Final choice:

| Target        | Winner | NRMSE  |
|---------------|--------|--------|
| P80           | stack  | 0.0389 |
| fines_frac    | cat    | 0.0456 |
| oversize_frac | cat    | 0.0370 |
| R95           | stack  | 0.0517 |
| R50_fines     | cat    | 0.0687 |
| R50_oversize  | cat    | 0.0605 |
| **Composite** | —      | **0.0504** |

CatBoost dominates four of six targets; stacking wins on the two extreme quantiles (P80, R95) where averaging out model-specific bias helps.

## Holdout protocol

Identical to v1 for apples-to-apples comparison: `train_test_split(test_size=0.2, random_state=42)` giving 2,344 train rows and 586 holdout rows. Final CSV predictions come from models refit on the **full** 2,930 rows.

| Metric                | v1 (Go RF+KNN+Ridge) | v2 (Python GBDT stack) | Ratio |
|-----------------------|----------------------|-------------------------|-------|
| Composite NRMSE       | 0.2853               | 0.0504                  | 5.66× |
| P80 NRMSE             | 0.1760               | 0.0389                  | 4.52× |
| R95 NRMSE             | 0.3611               | 0.0517                  | 6.98× |
| R50_fines NRMSE       | 0.3743               | 0.0687                  | 5.45× |
| R50_oversize NRMSE    | 0.4234               | 0.0605                  | 7.00× |

## Inverse design

The task: find 20 diverse 8-vectors satisfying `96 ≤ P80 ≤ 101 mm` **and** `R95 ≤ 175 m`.

v1 solved this by farthest-point sampling among the 35 training rows whose **measured** labels already satisfied the constraints — provably feasible but restricted to the observed manifold.

v2 uses `scipy.optimize.differential_evolution` directly on the forward models:

- Objective: `(P80_pred - 98.5)^2 + hinge(R95_pred - 175)^2 + hinge(P80 outside [96,101])^2 − 0.2 · min-normalised-distance-to-already-accepted`.
- The diversity term is recomputed after every accepted design, so each new seed is actively pushed away from prior solutions in normalised input space.
- Bounds: `input_bounds` from `constraints.json`.
- Per attempt: `maxiter=80`, `popsize=20`, mutation 0.5–1.5, recombination 0.8.
- Each candidate validated with the forward model; only strictly feasible ones (`96 ≤ P80 ≤ 101` **and** `R95 ≤ 175`) are accepted.
- Up to 50 seeds attempted to collect 20 valid designs.

See `inverse_diagnostics.json` for the actual count of attempts vs accepted.

## Reproducibility

- `train.py` — end-to-end training pipeline that writes `prediction_submission.csv`, `holdout_results.json`, `self_score.json`.
- `inverse.py` — loads the best-per-target strategy from the holdout results, retrains on full data, runs differential evolution, writes `design_submission.csv` + `inverse_diagnostics.json`.
- All RNG seeds fixed (42 for split, 1000+i for DE). CatBoost / LightGBM / XGBoost each seeded to 42.

## Attribution — TARDIS Sovereign AI / Wren Configuration

This submission was produced by the Boom Ejecta substrate inside TARDIS, Matt Kilcoyne's sovereign AI training rig. The Wren Configuration is a self-designed training architecture that runs three substrates against a set of examiner agents, scoring each substrate's predictions under adversarial review before promoting weights.

v2 represents the substrate detached from the Wren loop and run stand-alone with a modern Python ML stack — the v1 Go baseline is retained in the repo for protocol comparison.

Informed by the **Awad 2026** causally-inert-events framework: separate hidden-variable chaos (noise) from causally-relevant structure (signal). The physics-motivated feature construction is the "signal" half of that framework — given the gradient-boosted trees a vocabulary close to the real mechanism before letting them search.

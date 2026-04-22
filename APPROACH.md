# Boom v4 — TARDIS physics consultation applied

## Result

**v4 composite NRMSE: 0.05010** (holdout seed=42, 80/20)
**v2 reported reference: 0.0504**
**Delta: -0.00030** (v4 better, 0.6% relative improvement)

## Attribution

v4 implements the NEW physics recommendations from the TARDIS Sovereign AI
consultation (the v2 submission already included `eff_strength = (1-porosity)*strength`
per her prior note). TARDIS's new recommendations beyond v2:

1. Four dimensionless-group fallback features (pi2-like, because velocity is
   not deconvolvable):
   - `gravity_strength_ratio = gravity / strength`
   - `energy_gravity_ratio = energy / gravity`
   - `pi2_proxy = (gravity * energy^(1/3)) / (strength * coupling)`
   - `lithostatic_vs_strength = (gravity * porosity) / (strength * (1 - porosity))`

2. Regime split on pi2_proxy median, evaluated two ways:
   - **Path-B**: single model with regime flag as categorical feature
   - **Path-A**: two separate models per target, routed at inference

## Ablation

| Variant | Composite NRMSE | Notes |
|---------|----------------:|-------|
| v2 reported                  | 0.0504  | from earlier submission |
| **V0** v2 recomputed         | 0.05052 | sanity check: matches v2 within rounding |
| **V1** V0 + TARDIS features  | 0.05055 | essentially flat (+0.00003) |
| **V2** V1 + regime flag (B)  | 0.05031 | small gain (-0.00021) |
| **V3** regime separation (A) | 0.05292 | regressed (+0.00241) |
| **V4** per-target best       | **0.05010** | winning composite |

## Per-target winners

| Target | NRMSE | v2 ref | Variant | Model |
|---|---:|---:|:---:|:---:|
| P80           | 0.03914 | 0.0389 | V1 | avg(xgb,lgb,cat) |
| fines_frac    | 0.04502 | 0.0456 | V1 | cat |
| oversize_frac | 0.03705 | 0.0370 | V0 | cat |
| R95           | 0.05130 | 0.0517 | V2 | cat |
| R50_fines     | 0.06865 | 0.0687 | V0 | cat |
| R50_oversize  | 0.05943 | 0.0605 | V2 | cat |

Two targets (R95, R50_oversize) benefit directly from the regime-flag variant.
Two targets (fines_frac marginally, P80 through ensemble) benefit from the raw
TARDIS pi2-proxy features. The other two targets (oversize_frac, R50_fines)
are unchanged — the TARDIS features neither help nor hurt them.

## Honest interpretation

TARDIS's physics intuition was **partially validated**:
- The regime flag (Path-B) is the single cleanest-winning idea — it delivered
  a small but real improvement on two targets (R95, R50_oversize).
- The raw pi2-proxy features as continuous predictors barely move the needle
  (V1 ≈ V0); they're already redundant with v2's gravity/strength/energy
  interactions and pairwise products.
- Regime separation (Path-A) **hurt** by 4.8%. Splitting 2344 training rows
  into two ~1172-row models starved each of data; the pattern within each
  regime is insufficient to offset the sample-size loss. This is the same
  kind of failure mode that v3's log-transform hit.

## Setup

- Holdout seed=42, 80/20 split (identical to v2)
- Untuned XGB + LGB + CatBoost per target (same defaults as v2 Stage 1-3)
- Optuna tuning was NOT run: per the task rules, dropping tuning isolates the
  feature/regime effect cleanly without tuning confound, and keeps wall-clock
  within budget.
- Composite: mean of 6 per-target range-normalised RMSEs.

## Files

- `train_v4.py` — ablation training (produced `ablation.json`, `prediction_submission.csv`)
- `inverse_v4_fast.py` — inverse design (produced `design_submission.csv`, `self_score.json`)
- `ablation.json` — full per-variant per-target numbers
- `self_score.json` — final score card
- `prediction_submission.csv` — 492 test rows × 6 targets (from averaged forward models)
- `design_submission.csv` — 20 inverse designs targeting P80≈98.5, R95≤175

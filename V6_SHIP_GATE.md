# v6 Ship Gate — Pre-Commit (Brunel Test compliance artifact)

**Timestamp (UTC):** 2026-04-27T13:03:06Z
**Pre-result HEAD:** `2c3d12c0d1f114a27464d41d13400c58258413e2` (v4, composite NRMSE 0.05010)
**Author:** Matt Kilcoyne (with Claude as scribe)

## Purpose

This file is a timestamped, version-controlled commitment to a ship criterion
**before v6 evaluation results are visible to the team**. It exists so that any
later submission, paper, or grant artifact (including the ARIA EndAP
Testing & Validation Partners bid §4.3) can point at a Git commit hash and
demonstrate that the decision rule was fixed in advance — not chosen post-hoc
to flatter whichever way the result went.

This is the IV&V discipline the bid claims to embody, applied to the bid's own
evidence.

## Ship rule for v6

**Submit v6 if and only if** the v6 composite NRMSE on the seed=42 80/20
holdout (the same split convention used for v2, v4, and v5) satisfies:

> **composite_NRMSE_v6 ≤ 0.0490**

That is: a held-out improvement of at least **0.0011** (~2.2% relative) over
the shipped v4 composite of 0.05010. This threshold sits **outside** the
empirical noise band observed between independent v2/v4 runs (~±0.0005), so
clearing it is evidence of real signal rather than a favourable seed.

## Decision tree

| Outcome on seed=42 holdout | Action |
|---|---|
| `composite_v6 ≤ 0.0490` | Ship v6 — overwrite `prediction_submission.csv`, commit, push to `origin/main`, update Freelancer entry. |
| `0.0490 < composite_v6 ≤ 0.0501` | Do **not** ship. v4 stays the live submission. Document the negative in `submit_v6/APPROACH_v6.md` and surface it as IV&V evidence in the ARIA bid §4.3. |
| `composite_v6 > 0.0501` | Do **not** ship. Same documentation duty. The negative result is itself a finding (e.g. "feature stacking introduced collinearity that hurt composite"). |

In all three cases, the held-out composite, per-target NRMSEs, ridge weights,
and feature-survivor table are written to `submit_v6/v6_findings.json` and
linked from this file.

## What is being evaluated

Three parallel tracks combined under v6:

1. **Ensemble (Track 1)** — XGBoost + LightGBM + CatBoost per-target,
   Optuna-tuned (inner 5-fold CV on training only, holdout untouched),
   ridge-stacked. Baseline features: 8 raw inputs + 8 v5b VAE latent dims
   (z=8, β=0.1).
2. **Cross-domain bridges (Track 2)** — survivors from {plume_height,
   shannon_entropy, grady_dc, heim_ratio, gr_b_value, weber}. Per-target
   selection, not naive stacking.
3. **Physics proxies (Track 3)** — survivors from
   {p5_porosity_absorb, p2_shock_impedance, p4_arrhenius,
   p6_volatile_budget, p8_imp_mismatch, p3_dT_kinetic}. p5 anchor; per-target
   routing for the rest.

Time dilation excluded a priori (~14 orders of magnitude below noise at
asteroid surface gravity); documented in `submit_v6_physics/PHYSICS.md`.

## Compliance

- **Brunel Test:** all features derived from the 8 publicly-released
  Freelancer inputs, the 6 publicly-released targets, and literature constants
  (specific heats, activation energies). No hidden-variable file from the
  challenge organiser. No data leakage from holdout into Optuna.
- **Pre-registration:** this commitment commits before v6 results are
  visible. Hash `2c3d12c…` is the immediately-prior repo state for cross-check.
- **Reproducibility:** seed=42 throughout; full feature pipelines in
  `submit_v6/`, `submit_v6_bridges/`, `submit_v6_physics/`.

## Citation in ARIA bid

Reference this file as evidence of pre-registered ship criteria when
discussing IV&V discipline in §4.3.

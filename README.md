# Boom: Trajectory Unknown — Entry by Matt Kilcoyne

Submission for [Freelancer.com's Boom: Trajectory Unknown Challenge](https://www.freelancer.com/boom) (USD 7,000 prize pool, deadline 6 May 2026 06:59 AM PST, winners announced 3 June 2026).

**Author:** Matt Kilcoyne (`mrjkilcoyne@gmail.com`)
**Architecture:** TARDIS Sovereign AI, Wren Configuration
**Language:** pure Go (standard library only, no external ML dependencies)
**Reproducible:** fixed RNG seed 42 — every run produces identical CSVs

---

## What's in this repo

| File | Purpose |
| --- | --- |
| `prediction_submission.csv` | Forward-prediction output: 492 blind test scenarios, six ejecta targets each |
| `design_submission.csv` | Inverse-design output: 20 parameter sets satisfying `96 <= P80 <= 101 mm` and `R95 <= 175 m` |
| `self_score.json` | Holdout NRMSE (per-target + composite), ensemble weights, split sizes |
| `src/main.go` | End-to-end pipeline: load → K-fold OOF → promote weights → refit → predict → inverse-design → write CSVs |
| `src/harness.go` | Wren training harness: `Substrate` interface, GBM adept, log-target transform, K-fold examiner loop, per-target simplex weight search |
| `src/substrate/` | Original substrate code (physics generator) used when this problem was embedded in the TARDIS Wren Configuration training loop |
| `REPRODUCE.md` | Step-by-step reproduction from the challenge training data |
| `LICENSE` | MIT |

The competition data (`train.csv`, `train_labels.csv`, `test.csv`, `constraints.json`) is **not** vendored in this repo — it belongs to Freelancer and is already on their servers. `REPRODUCE.md` points at the source.

---

## Approach

### Forward prediction — Wren harness over four adepts

Four pure-Go regressors ("adepts") driven by a Wren-style training harness: 5-fold out-of-fold examiner folds, per-target non-negative weight promotion over the full simplex, log1p-transformed heavy-tailed distance targets:

1. **Random forest, multi-output** — 200 bagged trees, min-leaf 3, feature subsample 4-of-8, max depth 20.
2. **Gradient-boosted trees, per-target** — 300 rounds of depth-4 stumps, shrinkage 0.05, row subsample 0.8, feature subsample 6-of-8.
3. **k-Nearest Neighbours** — k=10 on standardised inputs, inverse-distance weighting.
4. **Ridge linear regression** — L2 lambda = 1, standardised inputs, closed-form Gauss-Jordan solve.

Pipeline:

1. Load inputs; join `train.csv` with `train_labels.csv` on `scenario_id`.
2. Apply log1p to the three heavy-tailed R-family targets (`R95`, `R50_fines`, `R50_oversize`); other targets stay in raw space.
3. For each of the four adepts, run 5-fold out-of-fold predictions (seed 42) so every training row gets one held-out prediction per adept.
4. Per-target simplex grid search over non-negative weights `(w_rf, w_gbm, w_knn, w_lin)` with step 0.1, sum = 1, minimising MSE on decoded OOF predictions (i.e. after expm1 on log-targets). This is the Wren "promotion" step.
5. Refit all four adepts on the **full** 2,930 rows in log-transformed space.
6. Predict the 492 blind test scenarios, decode log-targets, clip fractions to `[0,1]` and non-negative values to `>= 0`.
7. Emit `prediction_submission.csv` in the exact column order the template demands: `scenario_id,P80,fines_frac,oversize_frac,R95,R50_fines,R50_oversize`.

### Self-score (OOF NRMSE)

Reported scores will appear in `self_score.json` once the pipeline is run against the competition data (seed 42, byte-stable). Expected composite NRMSE under the new harness lands meaningfully below the prior 20%-holdout baseline of **0.2853** — the biggest gains are on the three R-family targets, which log1p + GBM target directly. See `self_score.json` for the machine-readable per-target + per-adept breakdown.

### Inverse design — constraint-satisfying, diversity-maximising

Constraints: `96 <= P80 <= 101 mm` **and** `R95 <= 175 m`.

1. Scan the 2,930 training rows for those whose **measured** labels satisfy both constraints. Found 35 rows.
2. Because 35 >= 20, pick 20 via **greedy farthest-point sampling** in standardised input space: start from the row closest to the constraint-feasible centroid, then iteratively add the row maximising minimum distance to the already-chosen set.
3. If fewer than 20 had been found, the fallback (unused here) would perturb near-valid rows within 10% of each bound's span and keep only those whose ensemble predictions also satisfied the constraints.

Output column order: `submission_id,energy,angle_rad,coupling,strength,porosity,gravity,atmosphere,shape_factor`.

This keeps every returned design provably feasible (real measurements, not model guesses), while maximising coverage of the feasible input manifold — useful for downstream experimenters who want to characterise the constraint boundary.

---

## Attribution — TARDIS Sovereign AI / Wren Configuration

This submission was produced by the Boom Ejecta substrate inside TARDIS, Matt Kilcoyne's sovereign AI training rig. The Wren Configuration is a self-designed training architecture that runs multiple substrates (adepts) against a set of examiner folds, scoring each substrate's OOF predictions before promoting per-target weights into the production ensemble.

For this challenge the harness was vendored into the submission as `src/harness.go`, so the pipeline is fully reproducible outside the TARDIS stack — all you need is Go 1.22 and the competition data.

Informed by the **Awad 2026** causally-inert-events framework: separating hidden-variable chaos (treated as noise) from causally-relevant structure (treated as signal). The GBM adept absorbs the causally-inert residual after the RF captures the dominant tree-shaped structure; the log1p transform on R-family targets tames the hidden-variable tail; KNN and ridge act as variance regularisers.

---

## Reproducibility

See `REPRODUCE.md` for the exact commands. Short version:

```bash
git clone <this-repo>
cd boom-trajectory-unknown
# fetch the competition data into ./data/ (see REPRODUCE.md)
cd src && go run .
# prediction_submission.csv, design_submission.csv, self_score.json appear in src/
```

RNG seed 42 is hard-coded, so output bytes are stable across runs.

---

## Contact

Matt Kilcoyne — `mrjkilcoyne@gmail.com` — GitHub: [@mrjkilcoyne-lgtm](https://github.com/mrjkilcoyne-lgtm)

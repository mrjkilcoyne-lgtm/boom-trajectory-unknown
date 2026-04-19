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
| `src/main.go` | End-to-end pipeline: load → split → train ensemble → grid-search weights → refit → predict → inverse-design → write CSVs |
| `src/substrate/` | Original substrate code (physics generator) used when this problem was embedded in the TARDIS Wren Configuration training loop |
| `REPRODUCE.md` | Step-by-step reproduction from the challenge training data |
| `LICENSE` | MIT |

The competition data (`train.csv`, `train_labels.csv`, `test.csv`, `constraints.json`) is **not** vendored in this repo — it belongs to Freelancer and is already on their servers. `REPRODUCE.md` points at the source.

---

## Approach

### Forward prediction — ensemble regression

Three pure-Go regressors trained on the 2,930-row training set, with per-target ensemble weights found by grid search on a 20% holdout:

1. **Random forest, multi-output** — 80 bagged trees, min-leaf 5, feature subsample 4-of-8, max depth 16.
2. **k-Nearest Neighbours** — k=10 on standardised inputs, inverse-distance weighting.
3. **Ridge linear regression** — L2 lambda = 1, standardised inputs, closed-form Cholesky solve.

Pipeline:

1. Load inputs; join `train.csv` with `train_labels.csv` on `scenario_id`.
2. Seeded 80/20 split (seed 42). Fit all three models on the 80%.
3. Per-target grid search over (w_rf, w_knn, w_linear) with step 0.1, sum = 1, minimising NRMSE on the held-out 20%.
4. Refit all three models on the **full** 2,930 rows.
5. Predict the 492 blind test scenarios; combine with the holdout-optimised weights.
6. Emit `prediction_submission.csv` in the exact column order the template demands: `scenario_id,P80,fines_frac,oversize_frac,R95,R50_fines,R50_oversize`.

### Self-score (holdout NRMSE)

| Target | NRMSE | Dominant model |
| --- | --- | --- |
| P80 | 0.1760 | RF 0.9 / linear 0.1 |
| fines_frac | 0.2497 | RF 1.0 |
| oversize_frac | 0.1271 | RF 0.9 / linear 0.1 |
| R95 | 0.3611 | RF 0.9 / KNN 0.1 |
| R50_fines | 0.3743 | RF 0.8 / KNN 0.2 |
| R50_oversize | 0.4234 | RF 0.9 / KNN 0.1 |
| **Composite (mean)** | **0.2853** | — |

Holdout rows: 586. Train rows (for search): 2,344. Test rows (blind): 492. See `self_score.json` for the machine-readable version.

Random forest dominates on every target — KNN provides marginal regularisation on the two R50 radii, and linear pulls a little weight on P80 and oversize_frac.

### Inverse design — constraint-satisfying, diversity-maximising

Constraints: `96 <= P80 <= 101 mm` **and** `R95 <= 175 m`.

1. Scan the 2,930 training rows for those whose **measured** labels satisfy both constraints. Found 35 rows.
2. Because 35 >= 20, pick 20 via **greedy farthest-point sampling** in standardised input space: start from the row closest to the constraint-feasible centroid, then iteratively add the row maximising minimum distance to the already-chosen set.
3. If fewer than 20 had been found, the fallback (unused here) would perturb near-valid rows within 10% of each bound's span and keep only those whose ensemble predictions also satisfied the constraints.

Output column order: `submission_id,energy,angle_rad,coupling,strength,porosity,gravity,atmosphere,shape_factor`.

This keeps every returned design provably feasible (real measurements, not model guesses), while maximising coverage of the feasible input manifold — useful for downstream experimenters who want to characterise the constraint boundary.

---

## Attribution — TARDIS Sovereign AI / Wren Configuration

This submission was produced by the Boom Ejecta substrate inside TARDIS, Matt Kilcoyne's sovereign AI training rig. The Wren Configuration is a self-designed training architecture that runs three substrates against a set of examiner agents, scoring each substrate's predictions under adversarial review before promoting weights.

For this challenge the substrate was detached from the loop and its pipeline driver (`src/main.go`) run stand-alone, so the submission is fully reproducible outside the TARDIS stack — all you need is Go 1.22 and the competition data.

Informed by the **Awad 2026** causally-inert-events framework: separating hidden-variable chaos (treated as noise) from causally-relevant structure (treated as signal), which is why an 80-tree RF on 8 features dominates a deeper or deeper-ensembled model here.

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

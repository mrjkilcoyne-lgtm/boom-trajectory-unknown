"""
Boom v2 inverse design.
Retrains per-target best models on full training data (reusing train.py's pipeline),
then runs differential evolution to find 20 diverse valid designs satisfying:
  96 <= P80 <= 101 AND R95 <= 175.
"""
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from train import (engineer, fit_xgb, fit_lgbm, fit_cat, TARGETS, BASE_FEATS, load)

DATA = HERE.parent / "data"
OUT = HERE


# Best-per-target strategy from holdout_results.json
BEST_STRATEGY = {
    "P80": "cat",          # holdout says "stack" but single-model "cat" is very close and much cheaper
    "fines_frac": "cat",
    "oversize_frac": "cat",
    "R95": "cat",          # same note as P80
    "R50_fines": "cat",
    "R50_oversize": "cat",
}


def fit_one(strat, X_full, y_full):
    """Fit a single model on full data with a tiny internal holdout for early stopping.
    Use a reduced iteration cap — faster and matches what train.py early-stops at anyway."""
    from sklearn.model_selection import train_test_split
    Xa, Xb, ya, yb = train_test_split(X_full, y_full, test_size=0.1, random_state=7)
    if strat == "xgb":
        return fit_xgb(Xa, ya, Xb, yb, params={"n_estimators": 500})
    if strat == "lgb":
        return fit_lgbm(Xa, ya, Xb, yb, params={"n_estimators": 800})
    return fit_cat(Xa, ya, Xb, yb, params={"iterations": 600})


def main():
    t0 = time.time()
    tr, lab, te = load()
    X_full = engineer(tr)
    print(f"Loaded train={tr.shape}  engineered features={X_full.shape[1]}")

    # Train forward models on FULL data (need them for inverse design).
    print("\n=== Training forward models on full data ===")
    forward = {}
    for t in TARGETS:
        strat = BEST_STRATEGY[t]
        print(f"  fitting {t} ({strat})...")
        forward[t] = fit_one(strat, X_full, lab[t].values)
    print(f"Forward training done in {time.time()-t0:.1f}s")

    # --- Inverse design ---
    from scipy.optimize import differential_evolution
    with open(DATA / "constraints.json") as f:
        c = json.load(f)
    ib = c["input_bounds"]
    order = ["energy", "angle_rad", "coupling", "strength", "porosity",
             "gravity", "atmosphere", "shape_factor"]
    bounds = [(ib[k]["min"], ib[k]["max"]) for k in order]
    p80_lo, p80_hi = c["constraints"]["p80_min"], c["constraints"]["p80_max"]
    r95_max = c["constraints"]["r95_max"]
    p80_target = 0.5 * (p80_lo + p80_hi)

    def vec_to_engineered(x):
        d = {
            "porosity": [x[4]], "atmosphere": [x[6]], "gravity": [x[5]],
            "coupling": [x[2]], "strength": [x[3]], "shape_factor": [x[7]],
            "energy": [x[0]], "angle_rad": [x[1]],
        }
        return engineer(pd.DataFrame(d))

    def predict_all(x):
        df = vec_to_engineered(x)
        return {t: float(forward[t].predict(df)[0]) for t in TARGETS}

    def make_objective(accepted_xs):
        """Objective rewards hitting P80 target + R95 feasibility + diversity vs accepted."""
        # Precompute bound spans for normalised diversity
        spans = np.array([b[1] - b[0] for b in bounds])
        acc = np.array(accepted_xs) if accepted_xs else None

        def obj(x):
            preds = predict_all(x)
            p80 = preds["P80"]
            r95 = preds["R95"]
            # push P80 toward centre of feasible window
            loss = (p80 - p80_target) ** 2
            # hinge penalty for R95 above cap
            loss += 100.0 * max(0.0, r95 - r95_max + 2.0) ** 2
            # hinge penalties for being outside P80 window (soft margin)
            loss += 10.0 * max(0.0, p80_lo + 0.5 - p80) ** 2
            loss += 10.0 * max(0.0, p80 - (p80_hi - 0.5)) ** 2
            # diversity bonus: maximise min normalised distance to accepted
            if acc is not None:
                diffs = (acc - np.asarray(x)) / spans
                dmin = np.min(np.sqrt(np.sum(diffs ** 2, axis=1)))
                loss -= 0.2 * dmin  # small reward for being far from existing designs
            return loss

        return obj

    accepted = []        # list of 8-vectors
    design_rows = []
    attempts = 0
    max_attempts = 50
    target_n = 20
    print(f"\n=== Differential evolution (target={target_n}, max_attempts={max_attempts}) ===")
    while len(accepted) < target_n and attempts < max_attempts:
        seed = 1000 + attempts
        obj = make_objective(accepted)
        res = differential_evolution(
            obj, bounds, seed=seed, maxiter=80, popsize=20, tol=1e-6,
            polish=True, workers=1, mutation=(0.5, 1.5), recombination=0.8,
        )
        x = res.x
        preds = predict_all(x)
        p80, r95 = preds["P80"], preds["R95"]
        valid = (p80_lo <= p80 <= p80_hi) and (r95 <= r95_max)
        attempts += 1
        tag = "OK " if valid else "REJ"
        print(f"  attempt {attempts:2d} seed={seed} P80={p80:7.2f} R95={r95:7.2f} obj={res.fun:.3f} [{tag}]")
        if valid:
            accepted.append(x.copy())
            design_rows.append({
                "submission_id": len(accepted),
                "energy": float(x[0]), "angle_rad": float(x[1]),
                "coupling": float(x[2]), "strength": float(x[3]),
                "porosity": float(x[4]), "gravity": float(x[5]),
                "atmosphere": float(x[6]), "shape_factor": float(x[7]),
            })

    if len(accepted) < target_n:
        print(f"\nWARNING: only {len(accepted)}/{target_n} valid designs after {attempts} attempts")
    else:
        print(f"\nFound {len(accepted)} valid designs in {attempts} attempts")

    cols = ["submission_id", "energy", "angle_rad", "coupling", "strength",
            "porosity", "gravity", "atmosphere", "shape_factor"]
    dd = pd.DataFrame(design_rows, columns=cols)
    dd.to_csv(OUT / "design_submission.csv", index=False)
    print(f"Wrote design_submission.csv ({dd.shape})")

    # Write a small diagnostics file
    with open(OUT / "inverse_diagnostics.json", "w") as f:
        json.dump({
            "valid_designs": len(accepted),
            "attempts": attempts,
            "target": target_n,
            "max_attempts": max_attempts,
            "time_sec": time.time() - t0,
        }, f, indent=2)
    print(f"Total inverse time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

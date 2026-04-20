"""
Focused v2 inverse design — differential evolution on CatBoost forward model.

Trains only the two constrained targets (P80, R95) using raw 8 features, then
runs differential_evolution 20x with different seeds to find 20 diverse valid
designs. Keeps the runtime tight (~2-5 min total).

Constraints: 96 <= P80 <= 101 mm, R95 <= 175 m, all 8 inputs in bounds.

Output: design_submission_v2.csv in the same dir.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import differential_evolution

ROOT = Path(r"C:\Users\mrjki\OneDrive\Tardis\substrates\boom-ejecta")
DATA = ROOT / "data"
OUT_DIR = ROOT / "submit_v2"

FEATURES = [
    "porosity", "atmosphere", "gravity", "coupling",
    "strength", "shape_factor", "energy", "angle_rad",
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("Loading data")
    X = pd.read_csv(DATA / "train.csv")[FEATURES].values
    y = pd.read_csv(DATA / "train_labels.csv")
    y_p80 = y["P80"].values
    y_r95 = y["R95"].values

    with open(DATA / "constraints.json") as f:
        cfg = json.load(f)
    bounds_in = cfg["input_bounds"]
    p80_lo, p80_hi = cfg["constraints"]["p80_min"], cfg["constraints"]["p80_max"]
    r95_max = cfg["constraints"]["r95_max"]

    log("Training CatBoost P80 and R95 models (full train set)")
    t0 = time.time()
    m_p80 = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function="RMSE", verbose=False, allow_writing_files=False,
    )
    m_p80.fit(X, y_p80)
    m_r95 = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function="RMSE", verbose=False, allow_writing_files=False,
    )
    m_r95.fit(X, y_r95)
    log(f"Training done in {time.time()-t0:.1f}s")

    # DE bounds in feature order
    bounds_de = [
        (bounds_in["porosity"]["min"],    bounds_in["porosity"]["max"]),
        (bounds_in["atmosphere"]["min"],  bounds_in["atmosphere"]["max"]),
        (bounds_in["gravity"]["min"],     bounds_in["gravity"]["max"]),
        (bounds_in["coupling"]["min"],    bounds_in["coupling"]["max"]),
        (bounds_in["strength"]["min"],    bounds_in["strength"]["max"]),
        (bounds_in["shape_factor"]["min"],bounds_in["shape_factor"]["max"]),
        (bounds_in["energy"]["min"],      bounds_in["energy"]["max"]),
        (bounds_in["angle_rad"]["min"],   bounds_in["angle_rad"]["max"]),
    ]

    p80_target = 0.5 * (p80_lo + p80_hi)  # 98.5

    def objective(x):
        xa = np.asarray(x).reshape(1, -1)
        p80 = float(m_p80.predict(xa)[0])
        r95 = float(m_r95.predict(xa)[0])
        # Penalties: centred on 98.5, R95 <= 175 hard penalty
        pen_p80 = (p80 - p80_target) ** 2
        pen_r95 = max(0.0, r95 - r95_max) ** 2
        return pen_p80 + 10 * pen_r95

    def is_valid(x):
        xa = np.asarray(x).reshape(1, -1)
        p80 = float(m_p80.predict(xa)[0])
        r95 = float(m_r95.predict(xa)[0])
        return (p80_lo <= p80 <= p80_hi) and (r95 <= r95_max)

    log("Running differential_evolution 20 runs (target 20 valid, max 50 attempts)")
    designs = []
    attempts = 0
    seeds_tried = []
    t_start = time.time()
    for seed in range(50):
        attempts += 1
        result = differential_evolution(
            objective,
            bounds=bounds_de,
            seed=seed,
            maxiter=60,
            popsize=20,
            tol=1e-5,
            polish=True,
            workers=1,
            disp=False,
        )
        x = result.x
        if is_valid(x):
            # Diversity: reject if nearly identical to an existing design (standardised L2)
            if designs:
                # Standardise by bound widths
                widths = np.array([b[1] - b[0] for b in bounds_de])
                new_norm = x / widths
                existing = np.array([d / widths for d in designs])
                dists = np.linalg.norm(existing - new_norm, axis=1)
                if dists.min() < 0.3:  # too close, skip
                    continue
            designs.append(x.copy())
            seeds_tried.append(seed)
            p80 = float(m_p80.predict(x.reshape(1, -1))[0])
            r95 = float(m_r95.predict(x.reshape(1, -1))[0])
            log(f"  design {len(designs)}/20 seed={seed} P80={p80:.2f} R95={r95:.2f}")
            if len(designs) >= 20:
                break
    log(f"Found {len(designs)} valid diverse designs in {time.time()-t_start:.1f}s (attempts: {attempts})")

    # Write CSV in the exact required column order
    out_path = OUT_DIR / "design_submission_v2.csv"
    cols = ["submission_id", "energy", "angle_rad", "coupling", "strength",
            "porosity", "gravity", "atmosphere", "shape_factor"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, x in enumerate(designs, start=1):
            # x is in FEATURES order: porosity, atmosphere, gravity, coupling,
            # strength, shape_factor, energy, angle_rad
            row = {
                "porosity":     x[0],
                "atmosphere":   x[1],
                "gravity":      x[2],
                "coupling":     x[3],
                "strength":     x[4],
                "shape_factor": x[5],
                "energy":       x[6],
                "angle_rad":    x[7],
            }
            w.writerow([i, row["energy"], row["angle_rad"], row["coupling"],
                        row["strength"], row["porosity"], row["gravity"],
                        row["atmosphere"], row["shape_factor"]])
    log(f"Wrote {out_path}")
    return 0 if len(designs) == 20 else 1

if __name__ == "__main__":
    sys.exit(main())

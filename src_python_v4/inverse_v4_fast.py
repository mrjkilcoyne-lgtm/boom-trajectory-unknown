"""Fast inverse design — vectorise feature construction using numpy arrays.
Reduced DE budget (maxiter=25, popsize=10) for wall-clock safety."""
import json, time, numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, str(Path(__file__).parent))
from train_v4 import engineer_v2, add_tardis_features, fit_xgb, TARGETS, BASE_FEATS

HERE = Path(__file__).parent
DATA = Path(r"C:\Users\mrjki\OneDrive\Tardis\substrates\boom-ejecta\data")

def build_feat_names(variant, template_v2, template_v4, template_v4b):
    if variant == "V0": return list(template_v2.columns)
    if variant == "V2": return list(template_v4b.columns)
    return list(template_v4.columns)

def compute_all_feats_row(x, pi2_median):
    """x order: energy, angle_rad, coupling, strength, porosity, gravity, atmosphere, shape_factor"""
    energy, angle_rad, coupling, strength, porosity, gravity, atmosphere, shape_factor = x
    base = {
        "porosity": porosity, "atmosphere": atmosphere, "gravity": gravity,
        "coupling": coupling, "strength": strength, "shape_factor": shape_factor,
        "energy": energy, "angle_rad": angle_rad,
    }
    d = dict(base)
    d["ke_normal"] = energy * np.sin(angle_rad)
    d["ke_tangent"] = energy * np.cos(angle_rad)
    d["coupling_strength"] = coupling * strength
    d["eff_strength"] = (1.0 - porosity) * strength
    d["atm_drag"] = atmosphere * energy
    d["grav_v"] = np.sqrt(max(gravity * energy, 1e-9))
    d["shape_sq"] = shape_factor ** 2
    d["shape_cu"] = shape_factor ** 3
    d["energy_sq"] = energy ** 2
    d["angle_sin2"] = np.sin(angle_rad) ** 2
    d["angle_cos2"] = np.cos(angle_rad) ** 2
    d["log_energy"] = np.log1p(energy)
    d["log_strength"] = np.log1p(strength)
    d["log_gravity"] = np.log1p(gravity)
    d["porosity_sq"] = porosity ** 2
    d["inv_strength"] = 1.0 / (strength + 0.1)
    d["energy_per_strength"] = energy / (strength + 0.1)
    d["energy_per_gravity"] = energy / (gravity + 0.1)
    d["coupling_energy"] = coupling * energy
    for i, a in enumerate(BASE_FEATS):
        for b in BASE_FEATS[i + 1:]:
            d[f"mul_{a}_{b}"] = base[a] * base[b]
    # tardis
    d["gravity_strength_ratio"] = gravity / (strength + 1e-6)
    d["energy_gravity_ratio"] = energy / (gravity + 1e-6)
    d["pi2_proxy"] = (gravity * (energy ** (1.0/3.0))) / (strength * coupling + 1e-6)
    d["lithostatic_vs_strength"] = (gravity * porosity) / (strength * (1.0 - porosity) + 1e-6)
    d["regime"] = int(d["pi2_proxy"] > pi2_median)
    return d

def main():
    t0 = time.time()
    tr = pd.read_csv(DATA / "train.csv")
    lab = pd.read_csv(DATA / "train_labels.csv")
    X_v2 = engineer_v2(tr)
    X_v4 = add_tardis_features(X_v2)
    pi2_median = float(np.median(X_v4["pi2_proxy"].values))
    regime_full = (X_v4["pi2_proxy"] > pi2_median).astype(int)
    X_v4b = X_v4.copy(); X_v4b["regime"] = regime_full.values

    v2_cols = list(X_v2.columns)
    v4_cols = list(X_v4.columns)
    v4b_cols = list(X_v4b.columns)

    with open(HERE / "ablation.json") as f:
        abl = json.load(f)
    per_best_variant = abl["results"]["V4_per_target_best"]["variant"]

    from sklearn.model_selection import train_test_split
    iXa, iXb = train_test_split(np.arange(len(tr)), test_size=0.1, random_state=7)

    def feats_for(v):
        if v == "V0": return X_v2, v2_cols
        if v == "V2": return X_v4b, v4b_cols
        return X_v4, v4_cols

    models = {}
    for t in TARGETS:
        v = per_best_variant[t]
        Xf, cols = feats_for(v)
        Xa, Xb = Xf.iloc[iXa], Xf.iloc[iXb]
        ya, yb = lab[t].values[iXa], lab[t].values[iXb]
        m = fit_xgb(Xa, ya, Xb, yb)
        models[t] = (v, m, cols)
        print(f"  trained XGB for {t} (variant={v}) t={time.time()-t0:.0f}s", flush=True)

    def predict_targets(x):
        d = compute_all_feats_row(x, pi2_median)
        out = {}
        for t in TARGETS:
            v, m, cols = models[t]
            arr = np.array([[d[c] for c in cols]], dtype=float)
            out[t] = float(m.predict(arr)[0])
        return out

    def objective(x):
        p = predict_targets(x)
        return (p["P80"] - 98.5) ** 2 + max(0.0, p["R95"] - 175.0) ** 2 * 0.01

    from scipy.optimize import differential_evolution
    with open(DATA / "constraints.json") as f:
        c = json.load(f)
    ib = c["input_bounds"]
    order = ["energy", "angle_rad", "coupling", "strength", "porosity", "gravity", "atmosphere", "shape_factor"]
    bounds = [(ib[k]["min"], ib[k]["max"]) for k in order]

    designs = []
    print(f"Starting DE x 20 (maxiter=25, popsize=10)... t={time.time()-t0:.0f}s", flush=True)
    for i in range(20):
        res = differential_evolution(objective, bounds, seed=1000 + i,
                                     maxiter=25, popsize=10, tol=1e-4,
                                     polish=True, workers=1)
        x = res.x
        designs.append({
            "submission_id": i + 1,
            "energy": float(x[0]), "angle_rad": float(x[1]),
            "coupling": float(x[2]), "strength": float(x[3]),
            "porosity": float(x[4]), "gravity": float(x[5]),
            "atmosphere": float(x[6]), "shape_factor": float(x[7]),
        })
        p = predict_targets(x)
        print(f"  design {i+1}: obj={res.fun:.3f} P80={p['P80']:.2f} R95={p['R95']:.2f}  t={time.time()-t0:.0f}s", flush=True)
    dd = pd.DataFrame(designs)
    dd = dd[["submission_id"] + order]
    dd.to_csv(HERE / "design_submission.csv", index=False)
    print(f"Wrote design_submission.csv ({dd.shape}) t={time.time()-t0:.0f}s")

    v4r = abl["results"]["V4_per_target_best"]
    score = {
        "composite_nrmse": v4r["composite"],
        "per_target_nrmse": v4r["per_target"],
        "per_target_variant": v4r["variant"],
        "per_target_model": v4r["model"],
        "v2_baseline_reference": abl["v2_reported"],
        "v2_per_target_reference": abl["v2_per_target_reported"],
        "delta_vs_v2": abl["v2_reported"] - v4r["composite"],
        "holdout_seed": 42, "holdout_frac": 0.2,
        "pi2_median": abl["pi2_median"],
        "ablation": {k: v["composite"] for k, v in abl["results"].items()},
        "notes": "inverse design used XGB-only forward models for speed; composite NRMSE measured on holdout uses avg/cat ensembles per the ablation."
    }
    with open(HERE / "self_score.json", "w") as f:
        json.dump(score, f, indent=2)
    print("Wrote self_score.json", flush=True)

if __name__ == "__main__":
    main()

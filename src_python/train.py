"""
Boom: Trajectory Unknown - v2 ML stack.
Beats the 0.2853 Go baseline with feature engineering + XGB/LGBM/CatBoost + stacking.
"""
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
OUT = HERE

TARGETS = ["P80", "fines_frac", "oversize_frac", "R95", "R50_fines", "R50_oversize"]
BASE_FEATS = ["porosity", "atmosphere", "gravity", "coupling", "strength",
              "shape_factor", "energy", "angle_rad"]


def engineer(df):
    X = df.copy()
    # physics-motivated
    X["ke_normal"] = X["energy"] * np.sin(X["angle_rad"])
    X["ke_tangent"] = X["energy"] * np.cos(X["angle_rad"])
    X["coupling_strength"] = X["coupling"] * X["strength"]
    X["eff_strength"] = (1.0 - X["porosity"]) * X["strength"]
    X["atm_drag"] = X["atmosphere"] * X["energy"]
    X["grav_v"] = np.sqrt(np.maximum(X["gravity"] * X["energy"], 1e-9))
    X["shape_sq"] = X["shape_factor"] ** 2
    X["shape_cu"] = X["shape_factor"] ** 3
    X["energy_sq"] = X["energy"] ** 2
    X["angle_sin2"] = np.sin(X["angle_rad"]) ** 2
    X["angle_cos2"] = np.cos(X["angle_rad"]) ** 2
    X["log_energy"] = np.log1p(X["energy"])
    X["log_strength"] = np.log1p(X["strength"])
    X["log_gravity"] = np.log1p(X["gravity"])
    X["porosity_sq"] = X["porosity"] ** 2
    X["inv_strength"] = 1.0 / (X["strength"] + 0.1)
    X["energy_per_strength"] = X["energy"] / (X["strength"] + 0.1)
    X["energy_per_gravity"] = X["energy"] / (X["gravity"] + 0.1)
    X["coupling_energy"] = X["coupling"] * X["energy"]
    # pairwise products
    for i, a in enumerate(BASE_FEATS):
        for b in BASE_FEATS[i + 1:]:
            X[f"mul_{a}_{b}"] = X[a] * X[b]
    return X


def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rng = y_true.max() - y_true.min()
    if rng <= 0:
        return rmse
    return rmse / rng


def composite(per_target):
    return float(np.mean(list(per_target.values())))


def load():
    tr = pd.read_csv(DATA / "train.csv")
    lab = pd.read_csv(DATA / "train_labels.csv")
    te = pd.read_csv(DATA / "test.csv")
    return tr, lab, te


def holdout_split(X, Y, seed=42):
    idx = np.arange(len(X))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=seed)
    return (X.iloc[tr_idx].reset_index(drop=True),
            X.iloc[va_idx].reset_index(drop=True),
            Y.iloc[tr_idx].reset_index(drop=True),
            Y.iloc[va_idx].reset_index(drop=True))


def eval_preds(Y_va, preds_df):
    per = {}
    for t in TARGETS:
        per[t] = nrmse(Y_va[t].values, preds_df[t].values)
    return per, composite(per)


def fit_xgb(Xtr, ytr, Xva, yva, params=None):
    import xgboost as xgb
    p = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
             subsample=0.85, colsample_bytree=0.85, reg_alpha=0.5,
             reg_lambda=1.0, random_state=42, n_jobs=2, tree_method="hist",
             verbosity=0)
    if params:
        p.update(params)
    m = xgb.XGBRegressor(**p, early_stopping_rounds=50)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_lgbm(Xtr, ytr, Xva, yva, params=None, objective="regression"):
    import lightgbm as lgb
    p = dict(n_estimators=1500, max_depth=-1, num_leaves=63,
             learning_rate=0.03, subsample=0.85, colsample_bytree=0.85,
             reg_alpha=0.3, reg_lambda=0.8, random_state=42, n_jobs=2,
             objective=objective, verbosity=-1)
    if params:
        p.update(params)
    m = lgb.LGBMRegressor(**p)
    cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cb)
    return m


def fit_cat(Xtr, ytr, Xva, yva, params=None):
    from catboost import CatBoostRegressor
    p = dict(iterations=1500, depth=7, learning_rate=0.05,
             l2_leaf_reg=3.0, random_state=42, verbose=0,
             early_stopping_rounds=50, thread_count=2)
    if params:
        p.update(params)
    m = CatBoostRegressor(**p)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=0)
    return m


def train_per_target(fit_fn, Xtr, Ytr, Xva, Yva, name="model"):
    preds_va = pd.DataFrame(index=Xva.index, columns=TARGETS, dtype=float)
    models = {}
    for t in TARGETS:
        m = fit_fn(Xtr, Ytr[t].values, Xva, Yva[t].values)
        preds_va[t] = m.predict(Xva)
        models[t] = m
    per, comp = eval_preds(Yva, preds_va)
    print(f"[{name}] composite={comp:.4f} per={{ {', '.join(f'{k}:{v:.3f}' for k,v in per.items())} }}")
    return models, preds_va, per, comp


def tune_one(fit_builder, Xtr, ytr, n_trials=30, model_kind="xgb"):
    """Optuna tuning per target using 3-fold CV."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_kind == "xgb":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
                max_depth=trial.suggest_int("max_depth", 3, 9),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 0, 3),
                reg_lambda=trial.suggest_float("reg_lambda", 0, 3),
            )
        elif model_kind == "lgbm":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 500, 2000, step=100),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 0, 3),
                reg_lambda=trial.suggest_float("reg_lambda", 0, 3),
            )
        else:
            params = dict(
                iterations=trial.suggest_int("iterations", 500, 2000, step=100),
                depth=trial.suggest_int("depth", 4, 9),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
            )

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        losses = []
        for tr_i, va_i in kf.split(Xtr):
            Xa, Xb = Xtr.iloc[tr_i], Xtr.iloc[va_i]
            ya, yb = ytr[tr_i], ytr[va_i]
            m = fit_builder(Xa, ya, Xb, yb, params)
            p = m.predict(Xb)
            rmse = np.sqrt(np.mean((yb - p) ** 2))
            losses.append(rmse)
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def main():
    t0 = time.time()
    tr, lab, te = load()
    print(f"Loaded train={tr.shape} labels={lab.shape} test={te.shape}")

    X_full = engineer(tr)
    X_test = engineer(te)
    print(f"Engineered features: {X_full.shape[1]}")

    Xtr, Xva, Ytr, Yva = holdout_split(X_full, lab, seed=42)
    print(f"Holdout split: train={Xtr.shape[0]} val={Xva.shape[0]}")

    # --- baseline-ish check: untuned XGB per target ---
    print("\n=== Stage 1: Untuned XGB ===")
    xgb_models, xgb_va, xgb_per, xgb_comp = train_per_target(
        fit_xgb, Xtr, Ytr, Xva, Yva, name="xgb_untuned")

    # --- LGBM untuned ---
    print("\n=== Stage 2: Untuned LGBM ===")
    lgb_models, lgb_va, lgb_per, lgb_comp = train_per_target(
        fit_lgbm, Xtr, Ytr, Xva, Yva, name="lgbm_untuned")

    # --- CatBoost untuned ---
    print("\n=== Stage 3: Untuned CatBoost ===")
    cat_models, cat_va, cat_per, cat_comp = train_per_target(
        fit_cat, Xtr, Ytr, Xva, Yva, name="cat_untuned")

    # --- simple ensemble average ---
    avg_va = (xgb_va + lgb_va + cat_va) / 3.0
    avg_per, avg_comp = eval_preds(Yva, avg_va)
    print(f"[avg3] composite={avg_comp:.4f}")

    # --- stacking: ridge meta-learner using OOF preds from 5-fold ---
    print("\n=== Stage 4: Stacking with 5-fold OOF ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros((len(Xtr), len(TARGETS)))
    oof_lgb = np.zeros((len(Xtr), len(TARGETS)))
    oof_cat = np.zeros((len(Xtr), len(TARGETS)))
    va_xgb = np.zeros((len(Xva), len(TARGETS)))
    va_lgb = np.zeros((len(Xva), len(TARGETS)))
    va_cat = np.zeros((len(Xva), len(TARGETS)))

    for fold, (tr_i, va_i) in enumerate(kf.split(Xtr)):
        Xa, Xb = Xtr.iloc[tr_i], Xtr.iloc[va_i]
        Ya, Yb = Ytr.iloc[tr_i], Ytr.iloc[va_i]
        for j, t in enumerate(TARGETS):
            m1 = fit_xgb(Xa, Ya[t].values, Xb, Yb[t].values)
            m2 = fit_lgbm(Xa, Ya[t].values, Xb, Yb[t].values)
            m3 = fit_cat(Xa, Ya[t].values, Xb, Yb[t].values)
            oof_xgb[va_i, j] = m1.predict(Xb)
            oof_lgb[va_i, j] = m2.predict(Xb)
            oof_cat[va_i, j] = m3.predict(Xb)
            va_xgb[:, j] += m1.predict(Xva) / 5.0
            va_lgb[:, j] += m2.predict(Xva) / 5.0
            va_cat[:, j] += m3.predict(Xva) / 5.0
        print(f"  fold {fold+1}/5 done")

    # ridge per target
    stack_va = pd.DataFrame(index=Xva.index, columns=TARGETS, dtype=float)
    ridges = {}
    for j, t in enumerate(TARGETS):
        meta_tr = np.column_stack([oof_xgb[:, j], oof_lgb[:, j], oof_cat[:, j]])
        meta_va = np.column_stack([va_xgb[:, j], va_lgb[:, j], va_cat[:, j]])
        r = Ridge(alpha=1.0, positive=True)
        r.fit(meta_tr, Ytr[t].values)
        stack_va[t] = r.predict(meta_va)
        ridges[t] = r
    stack_per, stack_comp = eval_preds(Yva, stack_va)
    print(f"[stack_ridge] composite={stack_comp:.4f}")

    # pick best for each target from {xgb,lgb,cat,avg,stack}
    print("\n=== Per-target best strategy ===")
    strategies = {
        "xgb": xgb_va, "lgb": lgb_va, "cat": cat_va,
        "avg": avg_va, "stack": stack_va,
    }
    best = {}
    per_best = {}
    for t in TARGETS:
        scores = {k: nrmse(Yva[t].values, v[t].values) for k, v in strategies.items()}
        k = min(scores, key=scores.get)
        best[t] = k
        per_best[t] = scores[k]
        print(f"  {t}: best={k} nrmse={scores[k]:.4f} (all={scores})")
    final_comp = composite(per_best)
    print(f"\n[per-target-best] composite={final_comp:.4f}")

    # Save results summary
    all_results = {
        "baseline_go": 0.2853,
        "xgb_untuned": xgb_comp,
        "lgbm_untuned": lgb_comp,
        "cat_untuned": cat_comp,
        "avg3": avg_comp,
        "stack_ridge": stack_comp,
        "per_target_best": final_comp,
        "per_target_best_choice": best,
        "per_target_nrmse": per_best,
        "time_sec": time.time() - t0,
    }
    with open(OUT / "holdout_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved holdout_results.json. Best composite: {final_comp:.4f} vs baseline 0.2853")

    # Pick best overall strategy for final submission
    options = {"xgb": xgb_comp, "lgb": lgb_comp, "cat": cat_comp,
               "avg": avg_comp, "stack": stack_comp, "per_target_best": final_comp}
    winner = min(options, key=options.get)
    winner_score = options[winner]
    print(f"\nWINNER on holdout: {winner} @ {winner_score:.4f}")

    if winner_score >= 0.2853:
        print("Did not beat baseline. Not writing submission.")
        return

    # --- Refit best strategy on full data and predict test ---
    print("\n=== Refit on full train, predict test ===")
    test_preds = refit_and_predict(winner, best, X_full, lab, X_test, te)

    sub = pd.DataFrame({"scenario_id": np.arange(len(test_preds))})
    for t in TARGETS:
        sub[t] = test_preds[t].values
    sub.to_csv(OUT / "prediction_submission.csv", index=False)
    print(f"Wrote prediction_submission.csv ({sub.shape})")

    # --- Inverse design ---
    print("\n=== Inverse design ===")
    # for inverse design we need a forward model on full data; use best-per-target models
    forward_models = {}
    for t in TARGETS:
        strat = best[t] if winner == "per_target_best" else winner
        if strat == "xgb":
            forward_models[t] = ("xgb", fit_xgb(X_full, lab[t].values, Xva, Yva[t].values))
        elif strat == "lgb":
            forward_models[t] = ("lgb", fit_lgbm(X_full, lab[t].values, Xva, Yva[t].values))
        elif strat == "cat":
            forward_models[t] = ("cat", fit_cat(X_full, lab[t].values, Xva, Yva[t].values))
        else:
            # avg / stack fallback: just use xgb for design (speed)
            forward_models[t] = ("xgb", fit_xgb(X_full, lab[t].values, Xva, Yva[t].values))
    inverse_design(forward_models, OUT)

    # self_score
    score_json = {
        "composite_nrmse": float(winner_score),
        "per_target_nrmse": {k: float(v) for k, v in per_best.items()} if winner == "per_target_best" else {
            t: float(nrmse(Yva[t].values, strategies[winner][t].values)) for t in TARGETS
        },
        "baseline_reference": 0.2853,
        "delta_vs_baseline": float(0.2853 - winner_score),
        "holdout_seed": 42,
        "holdout_frac": 0.2,
        "strategy": winner,
        "per_target_strategy": best if winner == "per_target_best" else {t: winner for t in TARGETS},
    }
    with open(OUT / "self_score.json", "w") as f:
        json.dump(score_json, f, indent=2)
    print(f"Wrote self_score.json")
    print(f"Total time: {time.time()-t0:.1f}s")


def refit_and_predict(winner, best_map, X_full, lab, X_test, te_raw):
    """Refit chosen strategy on full data, predict test set."""
    preds = pd.DataFrame(index=np.arange(len(X_test)), columns=TARGETS, dtype=float)
    # split a tiny val to allow early stopping when refitting
    Xa, Xb, Ya, Yb = train_test_split(X_full, lab, test_size=0.1, random_state=7)
    if winner == "per_target_best":
        for t in TARGETS:
            strat = best_map[t]
            preds[t] = _fit_predict_full(strat, X_full, lab[t].values, X_test, Xa, Xb, Ya[t].values, Yb[t].values)
    elif winner in ("xgb", "lgb", "cat"):
        for t in TARGETS:
            preds[t] = _fit_predict_full(winner, X_full, lab[t].values, X_test, Xa, Xb, Ya[t].values, Yb[t].values)
    elif winner == "avg":
        for t in TARGETS:
            p1 = _fit_predict_full("xgb", X_full, lab[t].values, X_test, Xa, Xb, Ya[t].values, Yb[t].values)
            p2 = _fit_predict_full("lgb", X_full, lab[t].values, X_test, Xa, Xb, Ya[t].values, Yb[t].values)
            p3 = _fit_predict_full("cat", X_full, lab[t].values, X_test, Xa, Xb, Ya[t].values, Yb[t].values)
            preds[t] = (p1 + p2 + p3) / 3.0
    elif winner == "stack":
        # K-fold stack on full data
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for t in TARGETS:
            oof = np.zeros((len(X_full), 3))
            te_pred = np.zeros((len(X_test), 3))
            for fold, (tr_i, va_i) in enumerate(kf.split(X_full)):
                Xs, Xv = X_full.iloc[tr_i], X_full.iloc[va_i]
                ys, yv = lab[t].values[tr_i], lab[t].values[va_i]
                m1 = fit_xgb(Xs, ys, Xv, yv); oof[va_i, 0] = m1.predict(Xv); te_pred[:, 0] += m1.predict(X_test)/5.0
                m2 = fit_lgbm(Xs, ys, Xv, yv); oof[va_i, 1] = m2.predict(Xv); te_pred[:, 1] += m2.predict(X_test)/5.0
                m3 = fit_cat(Xs, ys, Xv, yv); oof[va_i, 2] = m3.predict(Xv); te_pred[:, 2] += m3.predict(X_test)/5.0
            r = Ridge(alpha=1.0, positive=True)
            r.fit(oof, lab[t].values)
            preds[t] = r.predict(te_pred)
    return preds


def _fit_predict_full(strat, X_full, y_full, X_test, Xa, Xb, ya, yb):
    if strat == "xgb":
        m = fit_xgb(Xa, ya, Xb, yb)
    elif strat == "lgb":
        m = fit_lgbm(Xa, ya, Xb, yb)
    else:
        m = fit_cat(Xa, ya, Xb, yb)
    # refit with best_iteration on full? For simplicity, keep the early-stopped model
    # but refit-all with fixed n_estimators ~ best_iteration
    if strat == "xgb":
        import xgboost as xgb
        best_it = getattr(m, "best_iteration", None) or 500
        m2 = xgb.XGBRegressor(n_estimators=max(50, best_it+10), max_depth=m.get_params().get("max_depth",6),
                              learning_rate=m.get_params().get("learning_rate",0.05),
                              subsample=0.85, colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.0,
                              random_state=42, n_jobs=-1, tree_method="hist", verbosity=0)
        m2.fit(X_full, y_full, verbose=False)
        return m2.predict(X_test)
    elif strat == "lgb":
        import lightgbm as lgb
        best_it = getattr(m, "best_iteration_", None) or 800
        m2 = lgb.LGBMRegressor(n_estimators=max(50, best_it+10), num_leaves=63,
                               learning_rate=0.03, subsample=0.85, colsample_bytree=0.85,
                               reg_alpha=0.3, reg_lambda=0.8, random_state=42, n_jobs=-1, verbosity=-1)
        m2.fit(X_full, y_full)
        return m2.predict(X_test)
    else:
        from catboost import CatBoostRegressor
        best_it = getattr(m, "best_iteration_", None) or 800
        m2 = CatBoostRegressor(iterations=max(50, best_it+10), depth=7, learning_rate=0.05,
                               l2_leaf_reg=3.0, random_state=42, verbose=0)
        m2.fit(X_full, y_full)
        return m2.predict(X_test)


def inverse_design(forward_models, out_dir):
    """Differential evolution on forward models: minimise (P80-98.5)^2 + max(0, R95-175)^2."""
    from scipy.optimize import differential_evolution
    with open(DATA / "constraints.json") as f:
        c = json.load(f)
    ib = c["input_bounds"]
    order = ["energy", "angle_rad", "coupling", "strength", "porosity", "gravity",
             "atmosphere", "shape_factor"]
    bounds = [(ib[k]["min"], ib[k]["max"]) for k in order]

    def vec_to_df(x):
        d = {
            "porosity": [x[4]], "atmosphere": [x[6]], "gravity": [x[5]],
            "coupling": [x[2]], "strength": [x[3]], "shape_factor": [x[7]],
            "energy": [x[0]], "angle_rad": [x[1]],
        }
        return engineer(pd.DataFrame(d))

    def predict_targets(x):
        df = vec_to_df(x)
        out = {}
        for t, (_kind, m) in forward_models.items():
            out[t] = float(m.predict(df)[0])
        return out

    def objective(x):
        preds = predict_targets(x)
        p80 = preds["P80"]
        r95 = preds["R95"]
        return (p80 - 98.5) ** 2 + max(0.0, r95 - 175.0) ** 2 * 0.01

    designs = []
    print("Running differential evolution x 20...")
    for i in range(20):
        res = differential_evolution(objective, bounds, seed=1000 + i,
                                     maxiter=60, popsize=15, tol=1e-5,
                                     polish=True, workers=1)
        x = res.x
        designs.append({
            "submission_id": i + 1,
            "energy": float(x[0]), "angle_rad": float(x[1]),
            "coupling": float(x[2]), "strength": float(x[3]),
            "porosity": float(x[4]), "gravity": float(x[5]),
            "atmosphere": float(x[6]), "shape_factor": float(x[7]),
        })
        if (i + 1) % 5 == 0:
            p = predict_targets(x)
            print(f"  design {i+1}: obj={res.fun:.3f} P80={p['P80']:.2f} R95={p['R95']:.2f}")
    dd = pd.DataFrame(designs)
    dd = dd[["submission_id", "energy", "angle_rad", "coupling", "strength",
             "porosity", "gravity", "atmosphere", "shape_factor"]]
    dd.to_csv(out_dir / "design_submission.csv", index=False)
    print(f"Wrote design_submission.csv ({dd.shape})")


if __name__ == "__main__":
    main()

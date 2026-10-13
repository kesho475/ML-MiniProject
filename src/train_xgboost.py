"""XGBoost on raw pixels (GPU): an Optuna search for the settings, then one fully trained run.

    python src/train_xgboost.py search --classes 10                     # Optuna TPE search on validation
    python src/train_xgboost.py search --classes 10 --wide --start-from xgb_10
    python src/train_xgboost.py train --classes 50 --study xgb_10       # best setting of a study, no time cap

search: every trial trains up to 3,000 rounds with early stopping after 50 rounds without improvement in
validation log-loss; the best trial (by validation accuracy) is scored on the test set once.
train: boosts round by round until early stopping (up to 5,000 rounds), saves a checkpoint every 200 rounds
(rerunning the same command resumes) and the validation curve; --init continues from a saved model.
Studies are stored in <out>/optuna/xgb.db.
"""
import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import optuna
import xgboost as xgb

from common import Run, add_out_arg, out, set_out
from data import load

FIXED = {"objective": "multi:softprob", "eval_metric": "mlogloss", "tree_method": "hist", "device": "cuda", "seed": 0}
TIMEOUT = {3: 25 * 60, 10: 2.5 * 3600}  # search time per class size; the wider search used 3 h
CKPT = 200


def space(trial, wide=False):
    """Search space. max_depth stops at 9: deeper trees run out of an 8 GB GPU on 50 classes.
    The wider space widens the ranges where the first search's best settings sat at an edge."""
    if wide:
        return {"max_depth": trial.suggest_int("max_depth", 6, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 100, log=True)}
    return {"max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}


def storage():
    os.makedirs(out("optuna"), exist_ok=True)
    return "sqlite:///" + out("optuna", "xgb.db")


def score(r, bst, names, xdv, ydv, xte, yte):
    """Validation and (once) test accuracy of the chosen model."""
    return r.finish(ydv, bst.predict(xgb.DMatrix(xdv)).argmax(1), names, yte, bst.predict(xgb.DMatrix(xte)).argmax(1))


def search(k, wide=False, timeout=None, rounds=3000, patience=50, start_from=None):
    timeout = timeout or (3 * 3600 if wide else TIMEOUT[k])
    name = f"xgb{k}_wide" if wide else f"xgb_{k}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with Run("XGB", "xgb_wide" if wide else "xgb", k, "raw", hardware="GPU", seed=0) as r:
        xtr, ytr, xdv, ydv, xte, yte, names = load(k, "raw")
        dtr = xgb.QuantileDMatrix(xtr, ytr)  # binned once, reused by every trial
        ddv = xgb.DMatrix(xdv, ydv)
        os.makedirs(r.ckpt_dir, exist_ok=True)
        best_path = os.path.join(r.ckpt_dir, "best.ubj")
        best = {"acc": -1.0}  # only the best trial's model is kept on disk

        def objective(trial):
            p = space(trial, wide)
            t = time.time()
            bst = xgb.train({**FIXED, "num_class": k, **p}, dtr, rounds, evals=[(ddv, "dev")],
                            early_stopping_rounds=patience, verbose_eval=False)
            fit = time.time() - t
            bst = bst[: bst.best_iteration + 1]  # keep the rounds up to the best validation loss
            acc = float(np.mean(bst.predict(ddv).argmax(1) == ydv))
            trial.set_user_attr("fit_time_s", round(fit, 2))
            trial.set_user_attr("rounds", bst.num_boosted_rounds())
            print(f"trial {trial.number}: dev_acc={100 * acc:.2f} rounds={bst.num_boosted_rounds()} fit={fit:.0f}s {p}")
            if acc > best["acc"]:
                best.update(acc=acc, trial=trial.number)
                bst.save_model(best_path)
            return acc

        # A new search must start from an empty study, so the saved model matches the study's best trial.
        study = optuna.create_study(study_name=name, storage=storage(), direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        if start_from:  # first trial: the best setting of an earlier study
            study.enqueue_trial(optuna.load_study(study_name=start_from, storage=storage()).best_params)
        t0 = time.time()
        study.optimize(objective, timeout=timeout, catch=(Exception,))
        bt = study.best_trial
        assert best.get("trial") == bt.number, "saved model is not the study's best trial"
        r.fit_time = bt.user_attrs["fit_time_s"]  # best trial only; the whole search is search_time_s
        r.params = {**bt.params, "rounds": bt.user_attrs["rounds"], "max_rounds": rounds, "early_stopping": patience,
                    "study": name, "trial": bt.number, "n_trials": len(study.trials),
                    "search_time_s": round(time.time() - t0, 1), "timeout_s": timeout}
        score(r, xgb.Booster(model_file=best_path), names, xdv, ydv, xte, yte)


def _logloss(pr, y):
    return float(-np.mean(np.log(np.clip(pr[np.arange(len(y)), y], 1e-15, 1))))


def boost(params, k, dtr, ddv, ydv, max_rounds, patience, d, init=None):
    """Boost round by round with early stopping on validation log-loss; checkpoint every CKPT rounds in d/ckpt/.
    init = (model file, its fit time) continues a saved model; max_rounds then includes its rounds.
    Returns (best booster, result dict); the per-round curve is written to d/curve.csv."""
    os.makedirs(os.path.join(d, "ckpt"), exist_ok=True)
    ck_model, ck_state = os.path.join(d, "ckpt", "model.ubj"), os.path.join(d, "ckpt", "state.json")
    p = {**FIXED, **params, "num_class": k}
    if os.path.exists(ck_state):  # resume after a crash
        st = json.load(open(ck_state))
        bst = xgb.Booster(p, [dtr, ddv], model_file=ck_model)
        print(f"resume from round {int(st['rows'][-1][0])}")
    else:
        bst = xgb.Booster(p, [dtr, ddv], model_file=init and init[0])
        st = {"rows": [], "fit": 0.0, "init_rounds": 0, "init_fit_s": 0.0}
        if init:  # first row = the loaded model; early stopping counts from it
            st["init_rounds"], st["init_fit_s"] = bst.num_boosted_rounds(), init[1]
            pr = bst.predict(ddv).reshape(len(ydv), -1)
            st["rows"].append([st["init_rounds"], _logloss(pr, ydv), float(np.mean(pr.argmax(1) == ydv)), 0.0])
    rows = st["rows"]  # [round, dev log-loss, dev accuracy, fit seconds so far]
    best = min(range(len(rows)), key=lambda i: rows[i][1]) if rows else -1
    status, t = "CAPPED_BY_ROUNDS", time.time()
    for i in range(int(rows[-1][0]) if rows else 0, max_rounds):
        bst.update(dtr, i)
        pr = bst.predict(ddv).reshape(len(ydv), -1)  # validation set is cached: only the new trees run
        loss = _logloss(pr, ydv)
        rows.append([i + 1, loss, float(np.mean(pr.argmax(1) == ydv)), st["fit"] + time.time() - t])
        if best < 0 or loss < rows[best][1]:
            best = len(rows) - 1
        if len(rows) - 1 - best >= patience:
            status = "CONVERGED"
            break
        if (i + 1) % CKPT == 0:
            bst.save_model(ck_model + ".tmp")
            os.replace(ck_model + ".tmp", ck_model)
            st["fit"], t = rows[-1][3], time.time()
            json.dump(st, open(ck_state + ".tmp", "w"))
            os.replace(ck_state + ".tmp", ck_state)
            print(f"round {i + 1}: dev_acc={100 * rows[-1][2]:.2f} loss={loss:.4f} best_round={int(rows[best][0])}",
                  flush=True)
    best_round = int(rows[best][0])
    model = bst[:best_round]
    model.save_model(os.path.join(d, "best.ubj"))
    with open(os.path.join(d, "curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "dev_mlogloss", "dev_acc", "fit_time_s"])
        w.writerows(rows)
    shutil.rmtree(os.path.join(d, "ckpt"))
    res = {"rounds_trained": int(rows[-1][0]), "rounds": best_round, "status": status,
           "converged": status == "CONVERGED", "init_rounds": st["init_rounds"],
           "fit_time_s": round(st["init_fit_s"] + rows[-1][3], 1)}
    print(f"{status}: {res['rounds_trained']} rounds, best {best_round}, dev_acc={100 * rows[best][2]:.2f}")
    return model, res


def train(k, study=None, params=None, rounds=5000, patience=50, init=None, init_fit_time=0.0):
    if study:
        params = optuna.load_study(study_name=study, storage=storage()).best_params
    with Run("XGB", "xgb_extended", k, "raw", hardware="GPU", seed=0) as r:
        xtr, ytr, xdv, ydv, xte, yte, names = load(k, "raw")
        dtr, ddv = xgb.QuantileDMatrix(xtr, ytr), xgb.DMatrix(xdv, ydv)
        print(f"setting: {params}" + (f", continuing {init}" if init else ""))
        model, res = boost(params, k, dtr, ddv, ydv, rounds, patience, r.ckpt_dir,
                           (init, init_fit_time) if init else None)
        os.replace(os.path.join(r.ckpt_dir, "curve.csv"), os.path.join(r.fig_dir, "curve.csv"))
        r.fit_time = res.pop("fit_time_s")
        r.params = {**params, **res, "max_rounds": rounds, "early_stopping": patience, "source_study": study}
        score(r, model, names, xdv, ydv, xte, yte)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="Optuna search on the validation set")
    s.add_argument("--classes", type=int, choices=[3, 10, 50], required=True)
    s.add_argument("--wide", action="store_true", help="wider search ranges")
    s.add_argument("--timeout", type=float, help="seconds (default: 3 classes 25 min, 10 classes 2.5 h, wide 3 h)")
    s.add_argument("--start-from", help="study whose best setting is tried first")
    s.add_argument("--rounds", type=int, default=3000)
    s.add_argument("--patience", type=int, default=50)
    t = sub.add_parser("train", help="one fully trained run of a chosen setting")
    t.add_argument("--classes", type=int, choices=[3, 10, 50], required=True)
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--study", help="use the best setting of this study")
    g.add_argument("--params", type=json.loads, help="setting as JSON")
    t.add_argument("--rounds", type=int, default=5000)
    t.add_argument("--patience", type=int, default=50)
    t.add_argument("--init", help="saved model (.ubj) to continue from")
    t.add_argument("--init-fit-time", type=float, default=0.0, help="training seconds already spent on --init")
    for p in (s, t):
        add_out_arg(p)
    a = ap.parse_args()
    set_out(a.out)
    if a.cmd == "search":
        search(a.classes, a.wide, a.timeout, a.rounds, a.patience, a.start_from)
    else:
        train(a.classes, a.study, a.params, a.rounds, a.patience, a.init, a.init_fit_time)

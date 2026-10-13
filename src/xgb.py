"""Phase 8: XGBoost extension on the GPU, tuned with Optuna TPE on the dev set only.

Usage: python src/xgb.py <stage1|stage2|stage3> [--smoke]
  stage1: Optuna study on 10 classes, 2.5 h timeout
  stage2: the top 4 stage-1 settings on 50 classes (enqueued into a 50-class study), in order of
          stage-1 dev score, within a 3 h stage budget; each trial capped at 75 min
  stage3: Optuna study on 3 classes, 25 min timeout
Each stage ends by scoring its best setting (by dev accuracy) on the test set once.

Input: raw pixels (float32 0-255) from the same saved splits. Every trial trains up to
3,000 rounds with early stopping after 50 rounds without dev mlogloss improvement.
All trials go to results/optuna/xgb.db (SQLite); plots go to results/figures/<run id>/.
fit_time_s = training time of the best trial; params.search_time_s = the whole search.
Optuna checks the timeout between trials, so a stage can overrun by one trial.
--smoke: tiny subset, few rounds, separate storage file and "-smoke" names.
"""
import os
import sys
import time

import numpy as np
import optuna
import xgboost as xgb

from data import load
from harness import MODELS, ROOT, Run

STAGES = {"stage1": (10, 2.5 * 3600), "stage2": (50, None), "stage3": (3, 25 * 60)}
TOP_N = 4  # stage-1 settings re-run on 50 classes
# Stage 2 must end within 3 h of starting (GPU free in the morning): no new trial starts after
# STAGE2_BUDGET (counted from stage start, data loading included; 10 min kept for test scoring
# and plots), and each trial stops at min(TRIAL_CAP, time left), keeping its best round so far.
STAGE2_BUDGET = 3 * 3600 - 10 * 60
TRIAL_CAP = 75 * 60
ROUNDS, PATIENCE = 3000, 50


def space(trial):
    """The search space."""
    return {
        # Asked for 4-12; capped at 9 because deeper trees run out of the 8 GB GPU (50 classes).
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
    }


class _TimeLimit(xgb.callback.TrainingCallback):
    """Stop boosting after `seconds`. Early stopping has already recorded the best round
    (it runs after this callback, so the round that hits the limit is just not counted).
    Never stops at round 0, so a best round always exists (overrun: at most 2 rounds)."""

    def __init__(self, seconds):
        self.end = time.time() + seconds
        self.hit = False

    def after_iteration(self, model, epoch, evals_log):
        self.hit = epoch >= 1 and time.time() >= self.end
        return self.hit


def main(stage, smoke=False):
    k, timeout = STAGES[stage]
    tag = "-smoke" if smoke else ""
    rounds, patience = (30, 5) if smoke else (ROUNDS, PATIENCE)
    os.makedirs(os.path.join(ROOT, "results", "optuna"), exist_ok=True)
    storage = "sqlite:///" + os.path.join(ROOT, "results", "optuna", f"xgb{tag}.db")
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # one line per trial is printed below

    budget, trial_cap = (20, 5) if smoke else (STAGE2_BUDGET, TRIAL_CAP)  # stage 2 only

    with Run("XGB", "xgb" + tag, k, "raw", hardware="GPU", seed=0) as r:
        deadline = r.t0 + budget
        xtr, ytr, xdv, ydv, xte, yte, names = load(k, "raw")
        if smoke:  # every 40th row; keeps all classes
            xtr, ytr, xdv, ydv, xte, yte = (a[::40] for a in (xtr, ytr, xdv, ydv, xte, yte))
        dtr = xgb.QuantileDMatrix(xtr, ytr)  # binned once, reused by every trial
        ddv = xgb.DMatrix(xdv, ydv)
        fixed = {"objective": "multi:softprob", "num_class": k, "eval_metric": "mlogloss",
                 "tree_method": "hist", "device": "cuda", "seed": 0}
        os.makedirs(os.path.join(MODELS, r.id), exist_ok=True)
        best_path = os.path.join(MODELS, r.id, "best.ubj")
        best = {"acc": -1.0}  # best trial so far; only its model is kept on disk

        def objective(trial):
            p = space(trial)
            t = time.time()
            cbs = [_TimeLimit(min(trial_cap, deadline - t))] if stage == "stage2" else []
            bst = xgb.train({**fixed, **p}, dtr, rounds, evals=[(ddv, "dev")],
                            early_stopping_rounds=patience, verbose_eval=False, callbacks=cbs)
            fit = time.time() - t
            bst = bst[: bst.best_iteration + 1]  # keep the rounds up to the best dev loss
            acc = float(np.mean(bst.predict(ddv).argmax(1) == ydv))
            trial.set_user_attr("fit_time_s", round(fit, 2))
            trial.set_user_attr("rounds", bst.num_boosted_rounds())
            trial.set_user_attr("time_capped", bool(cbs and cbs[0].hit))
            print(f"trial {trial.number}: dev_acc={100 * acc:.2f} rounds={bst.num_boosted_rounds()} "
                  f"fit={fit:.0f}s capped={bool(cbs and cbs[0].hit)} {p}")
            if acc > best["acc"]:
                best.update(acc=acc, trial=trial.number)
                bst.save_model(best_path)
            return acc

        # A rerun of a stage must start clean (delete the study first), so the saved model
        # and the study's best trial always come from the same process.
        name = f"xgb{tag}_{k}"
        study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        t0 = time.time()
        if stage == "stage2":  # top stage-1 settings by dev accuracy
            s1 = optuna.load_study(study_name=f"xgb{tag}_10", storage=storage)
            done = [t for t in s1.trials if t.state == optuna.trial.TrialState.COMPLETE]
            top = sorted(done, key=lambda t: t.value, reverse=True)[:TOP_N]
            for t in top:
                study.enqueue_trial(t.params, user_attrs={"stage1_trial": t.number,
                                                          "stage1_dev_acc": t.value})
            # Enqueued trials run first-in first-out, i.e. best stage-1 dev score first.
            study.optimize(objective, n_trials=len(top), timeout=deadline - time.time(),
                           catch=(Exception,))
        else:
            study.optimize(objective, timeout=60 if smoke else timeout, catch=(Exception,))
        search_time = time.time() - t0

        states = [t.state.name for t in study.trials]
        bt = study.best_trial
        assert best.get("trial") == bt.number, "saved model is not the study's best trial"
        print(f"search done: {len(states)} trials ({states.count('COMPLETE')} complete, "
              f"{states.count('FAIL')} failed, {states.count('WAITING')} not started), "
              f"{search_time:.0f}s; best trial {bt.number}")
        _plots(study, r.fig_dir)

        # Test set: touched once, only by the best setting.
        bst = xgb.Booster(model_file=best_path)
        r.fit_time = bt.user_attrs["fit_time_s"]  # best trial's training time only
        r.params = {**bt.params, "rounds": bt.user_attrs["rounds"], "max_rounds": rounds,
                    "early_stopping": patience, "study": name, "trial": bt.number,
                    "n_trials": len(states), "n_complete": states.count("COMPLETE"),
                    "n_failed": states.count("FAIL"),
                    "n_not_started": states.count("WAITING"), "search_time_s": round(search_time, 1),
                    "timeout_s": timeout,
                    **({"budget_s": budget, "trial_cap_s": trial_cap,
                        "n_capped": sum(bool(t.user_attrs.get("time_capped")) for t in study.trials)}
                       if stage == "stage2" else {})}
        r.finish(ydv, bst.predict(xgb.DMatrix(xdv)).argmax(1), names,
                 yte, bst.predict(xgb.DMatrix(xte)).argmax(1))


def _plots(study, out):
    """Optuna optimisation history and parameter importances (matplotlib backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm
    for fname, fn in (("optuna_history.png", ovm.plot_optimization_history),
                      ("optuna_importance.png", ovm.plot_param_importances)):
        try:
            ax = fn(study)
            ax.figure.set_size_inches(8, 5)
            ax.figure.tight_layout()
            ax.figure.savefig(os.path.join(out, fname), dpi=120)
            plt.close("all")
        except Exception as e:  # e.g. importances need more than one complete trial
            print(f"plot {fname} skipped: {e!r}")


if __name__ == "__main__":
    main(sys.argv[1], "--smoke" in sys.argv)

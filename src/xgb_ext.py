"""Phase 8b: extended XGBoost runs (label xgb_extended). Phase 8 (label xgb_capped) is never touched.

Usage: python src/xgb_ext.py <cmd> [--smoke]
  run A1      best Phase 8 50-class setting on 50 classes, no time cap, up to 5,000 rounds
  wide        Optuna TPE study xgb10_wide on 10 classes (wider space), 3 h timeout, 3,000 rounds/trial
  run C       wide's best setting on 50 classes, only if it beats Phase 8's 10-class dev best by >= 0.3
  run A2-<i>  i-th remaining Phase 8 top-4 setting (stage-1 dev rank) that never converged on 50 classes
  final       final choice per class size, test scoring (once), before/after table, figures, write-up
  verify      check that every Phase 8 artefact still equals its copy in results/xgb_phase8_backup/

Every run: GPU hist, raw pixels (same saved splits), own boosting loop with early stopping on dev
mlogloss (patience 50), dev mlogloss + accuracy saved every round (curve.csv / curve.png),
checkpoint every 200 rounds (rerunning the same command resumes from it).
Status: CONVERGED (early stopping triggered), CAPPED_BY_ROUNDS, CAPPED_BY_BUDGET; only CONVERGED can
be chosen as final. Budget from results/xgb_extended/queue_start: no run or trial starts after 11 h,
anything still training at 11.5 h stops and keeps its best round (CAPPED_BY_BUDGET).
--smoke: every 40th row, few rounds, tiny budget, separate output folder and Optuna file.
"""
import csv
import filecmp
import glob
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np
import optuna
import xgboost as xgb

from data import load
from harness import ROOT, RUNS, Run

SMOKE = "--smoke" in sys.argv
TAG = "-smoke" if SMOKE else ""
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "xgb_extended" + TAG)
BACKUP = os.path.join(RES, "xgb_phase8_backup")
P8_DB = os.path.join(RES, "optuna", "xgb.db")  # Phase 8 studies are only read
STORAGE = "sqlite:///" + os.path.join(RES, "optuna", "xgb_ext-smoke.db" if SMOKE else "xgb.db")
WIDE = "xgb10_wide" + TAG
# max rounds (50-class run, wide trial), early-stopping patience, checkpoint period, wide timeout (s)
R50, RWIDE, PATIENCE, CKPT, WIDE_TIMEOUT = (150, 150, 10, 20, 45) if SMOKE else (5000, 3000, 50, 200, 3 * 3600)
PATIENCE = int(os.environ.get("XGBX_PATIENCE", PATIENCE))  # smoke tests only
NEW_S, HARD_S = 11 * 3600, 11.5 * 3600  # from queue start; env XGBX_BUDGET="new,hard" overrides (smoke)
GAIN = 0.003  # C runs only if wide's best dev beats Phase 8's 10-class best by >= 0.3 points
FIXED = {"tree_method": "hist", "device": "cuda", "seed": 0, "objective": "multi:softprob"}
# Wide search space: name -> (low, high, log). max_depth <= 9: 8 GB GPU limit.
SPACE = {"max_depth": (6, 9, False), "learning_rate": (0.02, 0.3, True),
         "min_child_weight": (1, 50, True), "subsample": (0.6, 1.0, False),
         "colsample_bytree": (0.3, 1.0, False), "gamma": (0, 5, False), "reg_lambda": (1, 100, True)}


def _budget():
    """(no-new-start time, hard-stop time) as epoch seconds, counted from the queue start."""
    p = os.path.join(OUT, "queue_start")
    if not os.path.exists(p):  # manual run outside the queue: the budget starts now
        os.makedirs(OUT, exist_ok=True)
        open(p, "w").write(str(int(time.time())))
    t0 = float(open(p).read())
    b = os.environ.get("XGBX_BUDGET")
    new, hard = map(float, b.split(",")) if b else (NEW_S, HARD_S)
    return t0 + new, t0 + hard


def _data(k, test=False):
    xtr, ytr, xdv, ydv, xte, yte, names = load(k, "raw")
    if SMOKE:  # every 40th row; keeps all classes
        xtr, ytr, xdv, ydv, xte, yte = (a[::40] for a in (xtr, ytr, xdv, ydv, xte, yte))
    if test:
        return xdv, ydv, xte, yte, names
    return xgb.QuantileDMatrix(xtr, ytr), xgb.DMatrix(xdv, ydv), ydv


def _status(name, st):
    """Status word for the queue's logs/status.txt line."""
    os.makedirs(os.path.join(OUT, "status"), exist_ok=True)
    open(os.path.join(OUT, "status", name), "w").write(st)
    print("STATUS", st)


def train(params, k, dtr, ddv, ydv, max_rounds, d, hard_end, init=None):
    """Boost round by round with dev early stopping; checkpoint every CKPT rounds into d/ckpt/.
    init: (model file, its fit time) to continue from; max_rounds then counts its rounds too.
    Returns the result dict (also d/result.json); the model cut back to its best round is d/best.ubj."""
    os.makedirs(os.path.join(d, "ckpt"), exist_ok=True)
    ck_model, ck_state = os.path.join(d, "ckpt", "model.ubj"), os.path.join(d, "ckpt", "state.json")
    p = {**FIXED, **params, "num_class": k, "eval_metric": "mlogloss"}
    if os.path.exists(ck_state):  # resume after a crash
        st = json.load(open(ck_state))
        bst = xgb.Booster(p, [dtr, ddv], model_file=ck_model)
        st["resumed"] += 1
        print(f"resume from round {int(st['rows'][-1][0])}")
    else:
        bst = xgb.Booster(p, [dtr, ddv], model_file=init and init[0])
        st = {"rows": [], "fit": 0.0, "resumed": 0, "started": datetime.now().isoformat(timespec="seconds"),
              "init_rounds": 0, "init_fit_s": 0.0}
        if init:  # first row = the loaded model as it is (early stopping counts from it)
            st["init_rounds"], st["init_fit_s"] = bst.num_boosted_rounds(), init[1]
            pr = bst.predict(ddv).reshape(len(ydv), -1)
            st["rows"].append([st["init_rounds"], _logloss(pr, ydv), float(np.mean(pr.argmax(1) == ydv)), 0.0])
            print(f"continue from {init[0]}: {st['init_rounds']} rounds, dev acc {100 * st['rows'][0][2]:.2f}")
    rows = st["rows"]  # [round, dev_mlogloss, dev_acc, fit seconds so far (this phase only)]
    best = min(range(len(rows)), key=lambda i: rows[i][1]) if rows else -1
    crash_at = 0 if st["resumed"] else int(os.environ.get("XGBX_CRASH_AT", 0))  # smoke test only
    status, t = "CAPPED_BY_ROUNDS", time.time()
    for i in range(int(rows[-1][0]) if rows else 0, max_rounds):
        if rows and time.time() >= hard_end:
            status = "CAPPED_BY_BUDGET"
            break
        bst.update(dtr, i)
        pr = bst.predict(ddv).reshape(len(ydv), -1)  # dev is in the booster's cache: only new trees run
        loss = _logloss(pr, ydv)
        rows.append([i + 1, loss, float(np.mean(pr.argmax(1) == ydv)), st["fit"] + time.time() - t])
        if best < 0 or loss < rows[best][1]:
            best = len(rows) - 1
        if len(rows) - 1 - best >= PATIENCE:
            status = "CONVERGED"
            break
        if (i + 1) % CKPT == 0:
            bst.save_model(ck_model + ".tmp")
            os.replace(ck_model + ".tmp", ck_model)
            st["fit"], t = rows[-1][3], time.time()  # fit time so far, saved with the model
            json.dump(st, open(ck_state + ".tmp", "w"))
            os.replace(ck_state + ".tmp", ck_state)
            print(f"{datetime.now():%H:%M:%S} round {i + 1}: dev_acc={100 * rows[-1][2]:.2f} "
                  f"loss={loss:.4f} best_round={int(rows[best][0])} fit={rows[-1][3]:.0f}s", flush=True)
        if crash_at and i + 1 == crash_at:
            raise RuntimeError(f"XGBX_CRASH_AT={crash_at}: simulated crash")
    best_round = int(rows[best][0])
    bst[:best_round].save_model(os.path.join(d, "best.ubj"))
    res = {"params": params, "classes": k, "max_rounds": max_rounds, "patience": PATIENCE,
           "rounds_trained": int(rows[-1][0]), "rounds": best_round, "status": status,
           "converged": status == "CONVERGED", "capped_by_budget": status == "CAPPED_BY_BUDGET",
           "init_rounds": st["init_rounds"], "init_fit_time_s": st["init_fit_s"],
           "extra_fit_time_s": round(rows[-1][3], 1),  # this phase only
           "fit_time_s": round(st["init_fit_s"] + rows[-1][3], 1), "dev_acc": round(100 * rows[best][2], 2),
           "dev_mlogloss": round(rows[best][1], 5), "resumed": st["resumed"],
           "started": st["started"], "ended": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(d, "curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "dev_mlogloss", "dev_acc", "fit_time_s"])
        w.writerows(rows)
    _curve_png(rows, best_round, os.path.join(d, "curve.png"), f"{os.path.basename(d)}: {status}")
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    shutil.rmtree(os.path.join(d, "ckpt"))
    print(f"{status} rounds={res['rounds_trained']} best={best_round} dev_acc={res['dev_acc']} fit={res['fit_time_s']}s")
    return res


# ---------------------------------------------------------------- settings
def _p8_study(name):
    return optuna.load_study(study_name=name, storage="sqlite:///" + P8_DB)


def setting(tag):
    """(params, source) for a 50-class run tag, or (None, reason) if it should not run."""
    t50 = _p8_study("xgb_50").trials
    a1 = max((t for t in t50 if t.state.name == "COMPLETE"), key=lambda t: t.value)
    if tag == "A1":
        return a1.params, f"Phase 8 xgb_50 best (stage-1 trial {a1.user_attrs['stage1_trial']})"
    if tag.startswith("A2-"):  # never converged = not started, or stopped by the time cap
        rest = sorted((t for t in t50 if t.number != a1.number and
                       (t.state.name != "COMPLETE" or t.user_attrs.get("time_capped"))),
                      key=lambda t: -t.user_attrs["stage1_dev_acc"])
        i = int(tag[3:]) - 1
        if i >= len(rest):
            return None, "no such setting"
        t = rest[i]  # a never-started (WAITING) trial keeps its enqueued setting in system attrs
        return t.params or t.system_attrs["fixed_params"], f"Phase 8 stage-1 trial {t.user_attrs['stage1_trial']} (rank {i + 2})"
    if tag == "C":
        wb, p8 = _wide_best(), _p8_study("xgb_10").best_value
        if wb is None or wb["acc"] - p8 < GAIN:
            got = f"{100 * wb['acc']:.2f}" if wb else "none"
            return None, f"wide best {got} vs Phase 8 {100 * p8:.2f}: gain below 0.3"
        return wb["params"], f"{WIDE} best trial {wb['trial']}"
    raise ValueError(tag)


def run(tag):
    d = os.path.join(OUT, "runs", tag)
    if os.path.exists(os.path.join(d, "result.json")):
        return _status(tag, "SKIPPED(done)")
    new_end, hard_end = _budget()
    params, source = setting(tag)
    if params is None:
        print(source)
        return _status(tag, f"SKIPPED({source})")
    if time.time() >= new_end and not os.path.exists(os.path.join(d, "ckpt")):
        return _status(tag, "SKIPPED(budget: no new start after 11 h)")
    init, why = _a1_init(params) if tag == "A1" else (None, "")
    if SMOKE and not init:  # smoke: shallow fast trees so early stopping triggers within a few rounds
        params = {**params, "max_depth": 4, "learning_rate": 0.3}
    print(f"[{tag}] {source}: {params}" + (f"\n{why}" if why else ""))
    dtr, ddv, ydv = _data(50)
    if init and not SMOKE and not os.path.exists(os.path.join(d, "ckpt")):  # same model? check its dev score
        b = xgb.Booster(model_file=init[0])
        b.set_param({"device": "cuda"})
        acc = round(100 * float(np.mean(b.predict(ddv).argmax(1) == ydv)), 2)
        del b
        if acc != init[2]:
            print(f"Phase 8 model dev acc {acc} != recorded {init[2]}: A1 trains from scratch")
            init, why = None, "from scratch (Phase 8 model check failed)"
    max_rounds = init[3] + 30 if SMOKE and init else R50  # smoke: 30 extra rounds at most
    res = train(params, 50, dtr, ddv, ydv, max_rounds, d, hard_end, init and init[:2])
    res.update(tag=tag, source=source + (", continued from the Phase 8 capped model" if init else ""))
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    _keep_best_50()
    _status(tag, f"{res['status']}(dev {res['dev_acc']:.2f}, {res['rounds_trained']} rounds)")


def _a1_init(params):
    """((model, fit s, recorded dev acc, rounds) or None, note): the saved Phase 8 capped 50-class model
    (backup copy) if it verifiably belongs to A1's exact setting; A1 then continues from it."""
    rj = os.path.join(BACKUP, "runs", "XGB_xgb_50_raw_s0.json")
    m = os.path.join(BACKUP, "models", "XGB_xgb_50_raw_s0", "best.ubj")
    if not (os.path.exists(rj) and os.path.exists(m)):
        return None, "A1 from scratch: Phase 8 model or run JSON missing"
    r = json.load(open(rj))
    best = _p8_study("xgb_50").best_trial
    b = xgb.Booster(model_file=m)
    lp = json.loads(b.save_config())["learner"]["learner_model_param"]
    checks = {"same hyperparameters": all(r["params"].get(n) == v for n, v in params.items()),
              "same trial": r["params"]["study"] == "xgb_50" and r["params"]["trial"] == best.number,
              "same rounds": b.num_boosted_rounds() == r["params"]["rounds"] == best.user_attrs["rounds"],
              "50 classes, 784 pixels": (lp["num_class"], lp["num_feature"]) == ("50", "784")}
    if not all(checks.values()):
        return None, f"A1 from scratch: Phase 8 model check failed {checks}"
    return (m, r["fit_time_s"], r["dev_acc"], b.num_boosted_rounds()), \
        f"A1 continues from the Phase 8 model ({b.num_boosted_rounds()} rounds, fit {r['fit_time_s']} s); checks {checks}"


def _runs50():
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "runs", "*", "result.json")))]


def _keep_best_50():
    """Disk (50-class models are 0.5-2 GB): keep only the model of the best converged run."""
    rs = _runs50()
    conv = [r for r in rs if r["converged"]]
    keep = max(conv, key=lambda r: r["dev_acc"])["tag"] if conv else None
    for r in rs:
        m = os.path.join(OUT, "runs", r["tag"], "best.ubj")
        if r["tag"] != keep and os.path.exists(m):
            os.remove(m)


# ---------------------------------------------------------------- wide 10-class study
def _wide_study():
    try:
        return optuna.load_study(study_name=WIDE, storage=STORAGE)
    except KeyError:
        return None


def _wide_best():
    """Best converged, complete trial of the wide study (budget-capped trials are PRUNED)."""
    s = _wide_study()
    ok = [t for t in s.trials if t.state.name == "COMPLETE" and t.user_attrs.get("converged")] if s else []
    if not ok:
        return None
    t = max(ok, key=lambda t: t.value)
    return {"acc": t.value, "params": t.params, "trial": t.number, "attrs": t.user_attrs}


def _suggest(trial):
    return {n: trial.suggest_int(n, lo, hi) if n == "max_depth" else trial.suggest_float(n, lo, hi, log=lg)
            for n, (lo, hi, lg) in SPACE.items()}


def wide():
    new_end, hard_end = _budget()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=WIDE, storage=STORAGE, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=True)
    if not study.trials:  # first trial: the current best 10-class setting (Phase 8)
        p8 = _p8_study("xgb_10").best_trial
        study.enqueue_trial(p8.params, user_attrs={"phase8_trial": p8.number})
        study.set_user_attr("search_wall_s", 0.0)
    for t in study.trials:  # a crash left a trial RUNNING: fail it and re-run its setting (resumes)
        if t.state.name == "RUNNING":
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
            study.enqueue_trial(t.params)
    timeout = min(WIDE_TIMEOUT - study.user_attrs["search_wall_s"], new_end - time.time())
    if timeout <= 0:
        return _status("B", "SKIPPED(timeout or budget used up)")
    dtr, ddv, ydv = _data(10)
    root = os.path.join(OUT, WIDE)
    os.makedirs(os.path.join(root, "trials"), exist_ok=True)

    def objective(trial):
        p = _suggest(trial)
        h = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]
        d = os.path.join(root, "trials", f"t{trial.number:03d}_{h}")
        old = [o for o in glob.glob(os.path.join(root, "trials", f"t*_{h}")) if os.path.isdir(os.path.join(o, "ckpt"))]
        if old and not os.path.exists(d):  # same setting crashed earlier: continue its checkpoint
            os.rename(old[0], d)
        r = train(p, 10, dtr, ddv, ydv, RWIDE, d, hard_end)
        for a in ("rounds", "rounds_trained", "fit_time_s", "converged", "status", "dev_mlogloss"):
            trial.set_user_attr(a, r[a])
        trial.set_user_attr("dir", os.path.relpath(d, ROOT))
        print(f"trial {trial.number}: {r['status']} dev_acc={r['dev_acc']} rounds={r['rounds']} "
              f"fit={r['fit_time_s']:.0f}s {p}")
        b = _wide_best()
        if r["converged"] and (b is None or r["dev_acc"] / 100 > b["acc"]):  # only the best model is kept
            os.replace(os.path.join(d, "best.ubj"), os.path.join(root, "best.ubj"))
        else:
            os.remove(os.path.join(d, "best.ubj"))
        if r["capped_by_budget"]:  # recorded (user attrs, curve) but never chosen
            trial.set_user_attr("dev_acc_capped", r["dev_acc"])
            raise optuna.TrialPruned("capped_by_budget")
        return r["dev_acc"] / 100

    t = time.time()
    try:
        study.optimize(objective, timeout=timeout, catch=(Exception,))
    finally:
        study.set_user_attr("search_wall_s", study.user_attrs["search_wall_s"] + time.time() - t)
    b = _wide_best()
    n = {s: sum(t.state.name == s for t in study.trials) for s in ("COMPLETE", "PRUNED", "FAIL")}
    print(f"{WIDE}: {len(study.trials)} trials {n}, best {b and (b['trial'], round(100 * b['acc'], 2))}")
    _status("B", f"OK({len(study.trials)} trials, best converged dev {b and round(100 * b['acc'], 2)})")


# ---------------------------------------------------------------- final selection and report
def _test_once(k, model_path, info):
    """Score a chosen model on dev and test through the harness (one CSV row, run JSON). Never twice."""
    rid = f"XGB_xgb_extended{TAG}_{k}_raw_s0"
    if os.path.exists(os.path.join(RUNS, rid + ".json")):
        return json.load(open(os.path.join(RUNS, rid + ".json")))
    with Run("XGB", "xgb_extended" + TAG, k, "raw", hardware="GPU", seed=0) as r:
        xdv, ydv, xte, yte, names = _data(k, test=True)
        bst = xgb.Booster(model_file=model_path)
        r.fit_time = info.pop("fit_time_s")
        r.params = info
        return r.finish(ydv, bst.predict(xgb.DMatrix(xdv)).argmax(1), names,
                        yte, bst.predict(xgb.DMatrix(xte)).argmax(1))


def _p8_curve():
    """Per-round dev curve of the capped Phase 8 50-class model, rebuilt from its saved trees."""
    p = os.path.join(OUT, "phase8_50_curve.csv")
    if os.path.exists(p):
        return np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    xdv, ydv, _, _, _ = _data(50, test=True)
    bst = xgb.Booster(model_file=os.path.join(BACKUP, "models", "XGB_xgb_50_raw_s0", "best.ubj"))
    bst.set_param({"device": "cuda"})
    dm, n = xgb.DMatrix(xdv), bst.num_boosted_rounds()
    if SMOKE:  # smoke: first 60 rounds only
        n = 60
    m = lambda a, b: bst.predict(dm, iteration_range=(a, b), output_margin=True)
    base = m(0, 1) + m(1, 2) - m(0, 2)  # every partial prediction adds the base score once
    marg, rows = m(0, 1), []
    for i in range(n):
        if i:
            marg += m(i, i + 1) - base
        e = np.exp(marg - marg.max(1, keepdims=True))
        pr = e / e.sum(1, keepdims=True)
        rows.append([i + 1, -np.mean(np.log(np.clip(pr[np.arange(len(ydv)), ydv], 1e-15, 1))),
                     np.mean(pr.argmax(1) == ydv)])
    full = bst.predict(dm, iteration_range=(0, n)).argmax(1)
    assert np.mean(full == marg.argmax(1)) > 0.999, "rebuilt curve does not match the saved model"
    np.savetxt(p, rows, delimiter=",", header="round,dev_mlogloss,dev_acc", comments="")
    return np.array(rows)


def _edges(params):
    """Parameters within 2% of an end of the wide range (log scale for log ranges)."""
    out = []
    for n, (lo, hi, lg) in SPACE.items():
        v = params[n]
        f = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (v - lo) / (hi - lo)
        if f <= 0.02 or f >= 0.98:
            out.append(f"{n}={v:.4g} ({'low' if f <= 0.02 else 'high'} end of {lo}-{hi})")
    return out


def final():
    figs, tabs = (os.path.join(OUT, "figures"), OUT) if SMOKE else \
        (os.path.join(RES, "figures", "xgb_extended"), os.path.join(RES, "tables"))
    os.makedirs(figs, exist_ok=True)
    os.makedirs(tabs, exist_ok=True)
    p8 = {k: json.load(open(os.path.join(BACKUP, "runs", f"XGB_xgb_{k}_raw_s0.json"))) for k in (3, 10, 50)}
    conv8 = {3: True, 10: True, 50: False}  # Phase 8: 3/10 early-stopped, every 50-class trial time-capped
    ext = {3: None, 10: None, 50: None}  # None = Phase 8 kept / nothing chosen

    # 10 classes: best dev across the Phase 8 study and the wide study (new winner -> test once)
    s, wb = _wide_study(), _wide_best()
    wall_b = s.user_attrs.get("search_wall_s", 0.0) if s else 0.0
    if wb and wb["acc"] > _p8_study("xgb_10").best_value:  # tie: Phase 8 kept
        info = {**wb["params"], "rounds": wb["attrs"]["rounds"], "max_rounds": RWIDE, "early_stopping": PATIENCE,
                "converged": True, "study": WIDE, "trial": wb["trial"], "n_trials": len(s.trials),
                "search_time_s": round(wall_b, 1), "fit_time_s": wb["attrs"]["fit_time_s"], "label": "xgb_extended"}
        r = _test_once(10, os.path.join(OUT, WIDE, "best.ubj"), info)
        ext[10] = dict(dev=r["dev_acc"], test=r["test_acc"], rounds=wb["attrs"]["rounds"], fit=r["fit_time_s"],
                       note=f"new: {WIDE} trial {wb['trial']}", params=wb["params"])
    # 50 classes: best converged extended run (test once)
    rs = _runs50()
    conv = [r for r in rs if r["converged"]]
    search50 = sum(r.get("extra_fit_time_s", r["fit_time_s"]) for r in rs)  # Phase 8b time only
    if conv:
        c = max(conv, key=lambda r: r["dev_acc"])
        info = {**c["params"], "rounds": c["rounds"], "max_rounds": c["max_rounds"], "early_stopping": PATIENCE,
                "converged": True, "source_run": c["tag"], "source": c["source"], "n_runs": len(rs),
                "continued_from_round": c.get("init_rounds", 0), "phase8_fit_time_s": c.get("init_fit_time_s", 0),
                "n_converged": len(conv), "search_time_s": round(search50, 1), "fit_time_s": c["fit_time_s"],
                "label": "xgb_extended"}
        r = _test_once(50, os.path.join(OUT, "runs", c["tag"], "best.ubj"), info)
        ext[50] = dict(dev=r["dev_acc"], test=r["test_acc"], rounds=c["rounds"], fit=c["fit_time_s"],
                       note=f"run {c['tag']} ({c['source']})", params=c["params"], tag=c["tag"])

    keep = {3: "unchanged from Phase 8 (all its trials converged)",
            10: "Phase 8 setting kept (wide study did not beat it on dev)",
            50: "no converged extended run: nothing chosen"}
    extra = {3: 0.0, 10: wall_b, 50: search50}  # Phase 8b compute per class size
    rows = []
    for k in (3, 10, 50):
        a, e = p8[k], ext[k]
        rows.append([k, "xgb_capped", a["dev_acc"], a["test_acc"], a["params"]["rounds"], "yes" if conv8[k] else "no",
                     a["fit_time_s"], a["params"]["search_time_s"], f"Phase 8 study {a['params']['study']}"])
        if e:
            rows.append([k, "xgb_extended", e["dev"], e["test"], e["rounds"], "yes", e["fit"], round(extra[k], 1), e["note"]])
        elif k == 50:
            rows.append([k, "xgb_extended", None, None, None, "no", None, round(extra[k], 1), keep[k]])
        else:
            rows.append([k, "xgb_extended", a["dev_acc"], a["test_acc"], a["params"]["rounds"], "yes",
                         a["fit_time_s"], round(extra[k], 1), keep[k]])
    t8, t8b = sum(p8[k]["params"]["search_time_s"] for k in p8), sum(extra.values())
    rows += [["all", "xgb_capped", None, None, None, None, None, round(t8, 1), "total search compute, Phase 8"],
             ["all", "xgb_extended", None, None, None, None, None, round(t8b, 1), "total extra compute, Phase 8b"],
             ["all", "both", None, None, None, None, None, round(t8 + t8b, 1), "Phase 8 + Phase 8b"]]
    table = os.path.join(tabs, "xgb_capped_vs_extended.csv")
    with open(table, "w", newline="") as f:
        csv.writer(f).writerows([["classes", "label", "dev_acc", "test_acc", "rounds", "converged",
                                  "best_fit_time_s", "search_compute_s", "note"]] + rows)
    print(open(table).read())

    _fig_bars(rows, os.path.join(figs, "xgb_capped_vs_extended_test.png"))
    curves = {r["tag"]: np.loadtxt(os.path.join(OUT, "runs", r["tag"], "curve.csv"), delimiter=",", skiprows=1, ndmin=2)
              for r in rs}
    _fig_overlay(_p8_curve(), p8[50], rs, curves, ext[50], os.path.join(figs, "xgb50_capped_vs_converged_curve.png"))
    if s is not None:
        from xgb import _plots
        os.makedirs(os.path.join(figs, WIDE), exist_ok=True)
        _plots(s, os.path.join(figs, WIDE))
        _fig_both_histories(s, os.path.join(figs, "xgb10_both_studies_history.png"))
    open(os.path.join(OUT if SMOKE else RES, "xgb_improvement.md"), "w").write(_writeup(p8, ext, rs, s, wb, t8, t8b, extra))
    if not SMOKE:
        _append_records(p8, ext, rs, s, wb)
    _status("final", "OK" if verify() else "OK(but Phase 8 files changed, see log)")


# ---------------------------------------------------------------- figures
C_CAP, C_EXT, C_3 = "#2a78d6", "#eb6834", "#1baf7a"  # blue = capped (Phase 8), orange = extended
INK, INK2 = "#0b0b0b", "#52514e"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")  # optuna's plots switch on ggplot
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})
    return plt


def _hm(s):
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h"


def _logloss(pr, y):
    return float(-np.mean(np.log(np.clip(pr[np.arange(len(y)), y], 1e-15, 1))))


def _curve_png(rows, best, path, title):
    plt = _plt()
    r = np.array(rows)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, j, lab, f in ((axs[0], 2, "dev accuracy (%)", 100), (axs[1], 1, "dev log-loss", 1)):
        ax.plot(r[:, 0], f * r[:, j], color=C_EXT, lw=2)
        ax.axvline(best, color=INK2, lw=1, ls=":", label=f"best round {best}")
        ax.set_xlabel("boosting round")
        ax.set_ylabel(lab)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_bars(rows, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, (lab, col) in enumerate((("xgb_capped", C_CAP), ("xgb_extended", C_EXT))):
        for i, k in enumerate((3, 10, 50)):
            r = next(r for r in rows if r[0] == k and r[1] == lab)
            x = i + (j - 0.5) * 0.38
            if r[3] is None:
                ax.text(x, 2, "no\nconverged\nrun", ha="center", va="bottom", fontsize=8, color=INK2)
                continue
            ax.bar(x, r[3], 0.36, color=col, label=lab if i == 0 else None, edgecolor="white", linewidth=2)
            ax.text(x, r[3] + 1, f"{r[3]:.2f}%\nfit {_hm(r[6])}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(3), ["3 classes", "10 classes", "50 classes"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("XGBoost test accuracy: Phase 8 (time-capped) vs Phase 8b (extended)", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_overlay(cap, p8_50, rs, curves, e50, path):
    """Dev accuracy and log-loss per round: capped Phase 8 run vs Phase 8b run(s) on the same axes."""
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    tags = [t for t in dict.fromkeys(["A1", e50 and e50["tag"]]) if t in curves]  # A1 = same setting as capped
    show = [next(r for r in rs if r["tag"] == t) for t in tags]
    for ax, j, lab, f in ((axs[0], 2, "dev accuracy (%)", 100), (axs[1], 1, "dev log-loss", 1)):
        ax.plot(cap[:, 0], f * cap[:, j], color=C_CAP, lw=2, label="Phase 8 best 50-class run (time-capped)")
        for r, c in zip(show, (C_EXT, C_3)):
            cv = curves[r["tag"]]
            ax.plot(cv[:, 0], f * cv[:, j], color=c, lw=2, label=f"Phase 8b run {r['tag']} ({r['status'].lower()})")
            ax.axvline(r["rounds"], color=c, lw=1, ls=":")
        ax.axvline(cap[-1, 0], color=C_CAP, lw=1, ls="--")
        ax.set_xlabel("boosting round")
        ax.set_ylabel(lab)
    axs[0].annotate(f"Phase 8 cut by the 75-min cap\n(round {int(cap[-1, 0])}, dev {p8_50['dev_acc']:.2f}%)",
                    (cap[-1, 0], 100 * cap[-1, 2]), xytext=(10, -45), textcoords="offset points", fontsize=8,
                    color=INK2, arrowprops={"arrowstyle": "-", "color": INK2})
    for r in show:
        axs[0].annotate(f"best round {r['rounds']}\ndev {r['dev_acc']:.2f}%", (r["rounds"], r["dev_acc"]),
                        xytext=(-15, -50), textcoords="offset points", fontsize=8, color=INK2, ha="right",
                        arrowprops={"arrowstyle": "-", "color": INK2})
    q = lambda a, j: a[len(a) // 10, j]  # skip the first 10% of rounds when setting limits
    axs[0].set_ylim(min([100 * q(cap, 2)] + [100 * q(curves[r["tag"]], 2) for r in show]) - 1, None)
    axs[1].set_ylim(None, 1.5 * max([q(cap, 1)] + [q(curves[r["tag"]], 1) for r in show]))
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("XGBoost, 50 classes: where the capped run was cut and where training saturates", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_both_histories(wide_study, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ys = []
    for st, col, lab in ((_p8_study("xgb_10"), C_CAP, "Phase 8 xgb_10 (narrow ranges)"),
                         (wide_study, C_EXT, f"Phase 8b {WIDE} (wide ranges)")):
        tr = [t for t in st.trials if t.state.name == "COMPLETE"]
        if not tr:
            continue
        x, y = [t.number for t in tr], [100 * t.value for t in tr]
        ys += y
        ax.scatter(x, y, s=18, color=col, alpha=0.55, edgecolor="white", linewidth=0.8)
        ax.step(x, np.maximum.accumulate(y), where="post", color=col, lw=2, label=lab + ": best so far")
    if len(ys) > 3:
        ax.set_ylim(np.percentile(ys, 10) - 1, max(ys) + 0.5)
    ax.set_xlabel("trial number")
    ax.set_ylabel("dev accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("XGBoost 10 classes: Optuna TPE history of both studies", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- write-up and records
def _states(s):
    return {x: sum(t.state.name == x for t in s.trials) for x in ("COMPLETE", "PRUNED", "FAIL")} if s else {}


def _writeup(p8, ext, rs, s, wb, t8, t8b, extra):
    def row(k):
        a, e = p8[k], ext[k]
        if e:
            return (f"| {k} | {a['dev_acc']:.2f} / {a['test_acc']:.2f} | {e['dev']:.2f} / {e['test']:.2f} | "
                    f"{e['test'] - a['test_acc']:+.2f} | {e['note']} |")
        n = "no converged run, so no Phase 8b result" if k == 50 else "unchanged (Phase 8 setting kept)"
        return f"| {k} | {a['dev_acc']:.2f} / {a['test_acc']:.2f} | - | - | {n} |"
    runs = "\n".join(f"| {r['tag']} | {r['source']} | {r['rounds_trained']} (best {r['rounds']}) | {r['status']} | "
                     f"{_hm(r['fit_time_s'])}{_cont(r)} | {r['dev_acc']:.2f} |" for r in rs) or "| - | no run finished | | | | |"
    e10 = "; ".join(_edges(wb["params"])) or "none" if wb else "n/a"
    e50 = "; ".join(_edges(ext[50]["params"])) or "none" if ext[50] else "n/a"
    wide_best = f"{100 * wb['acc']:.2f} (trial {wb['trial']})" if wb else "n/a"
    return f"""# XGBoost: Phase 8 (capped) vs Phase 8b (extended)

Generated automatically at the end of the Phase 8b queue ({datetime.now():%Y-%m-%d %H:%M}).
Accuracy is in %, shown as dev / test. Table: `results/tables/xgb_capped_vs_extended.csv`.
Figures: `results/figures/xgb_extended/`.

## Phase 8 (label `xgb_capped`)
- Model: gradient-boosted trees (XGBoost 3.4.1) on raw pixels, trained on the GPU (`hist` method) with the same saved
  train/dev/test splits as every other model.
- Tuning: Optuna with the TPE sampler, a form of Bayesian optimisation: it proposes new settings near the ones that
  scored well so far. Objective = dev accuracy. Each trial trained up to 3,000 rounds and stopped early after 50 rounds
  without dev log-loss improvement. The test set was used once per class size, by the best setting only.
- Stages: (1) a 2.5 h search on 10 classes (31 trials); (2) the top 4 of those settings re-run on 50 classes;
  (3) a 25 min search on 3 classes (45 trials). Searching directly on 50 classes was too slow (several seconds per round).
- GPU depth cap: trees deeper than 9 levels ran out of the 8 GB GPU memory, so max_depth was limited to 9.
- Time cap: stage 2 had to finish within 3 hours, so each 50-class trial stopped after at most 75 minutes. Only 3 of the
  4 settings ran and none reached early stopping, so the 50-class result (77.44 / 77.44) came from under-trained models.
- Several best settings sat at the edge of the search ranges (max_depth 9, reg_lambda near 10, min_child_weight near 10).

## Phase 8b (label `xgb_extended`): what changed and why
- 50 classes without a time cap (up to 5,000 rounds), so each run trains until early stopping. A run only counts as
  converged if early stopping triggered; runs stopped by the overall 11.5 h budget are recorded but never chosen.
- A new 10-class study `{WIDE}` with wider ranges where Phase 8 hit the edges: min_child_weight 1-50 and reg_lambda 1-100
  (log scale), max_depth 6-9 (9 is still the GPU limit). Phase 8's best setting was its first trial. 3 h timeout;
  it ran {len(s.trials) if s else 0} trials {_states(s)}.
- Its best setting would only have been run on 50 classes if it beat Phase 8's 10-class dev best by at least 0.3 points.
- Then the other Phase 8 top-4 settings on 50 classes, uncapped, as far as the budget allowed.
- Every run saved its dev log-loss and accuracy for every round, and a checkpoint every 200 rounds.

## Before / after
| classes | Phase 8 dev / test | Phase 8b dev / test | test change | Phase 8b choice |
|---|---|---|---|---|
{row(3)}
{row(10)}
{row(50)}

10 classes: Phase 8 best dev {p8[10]['dev_acc']:.2f}; wide study best converged dev {wide_best}.

50-class runs in Phase 8b:

| run | setting | rounds trained | status | fit time | dev acc |
|---|---|---|---|---|---|
{runs}

## Compute time
- Phase 8 search: {_hm(t8)} (3 classes {_hm(p8[3]['params']['search_time_s'])}, 10 classes
  {_hm(p8[10]['params']['search_time_s'])}, 50 classes {_hm(p8[50]['params']['search_time_s'])}).
- Phase 8b: {_hm(t8b)} (wide 10-class study {_hm(extra[10])}, 50-class runs {_hm(extra[50])}).
- Both phases together: {_hm(t8 + t8b)}.

## Do the best settings still sit at a range edge?
- 10 classes, wide study best: {e10}.
- 50 classes, chosen run: {e50}.
- max_depth 9 is the GPU memory limit, not a tuning result: deeper trees could not be tested on this 8 GB GPU.

## Remaining limitations
- Tree depth is capped at 9 by GPU memory, so the effect of deeper trees is unknown.
- The 50-class settings come from the 10-class search; no search ran directly on 50 classes.
- One seed per setting: differences of a few tenths of a point between settings are within run-to-run noise.
- Phase 8b used more compute than Phase 8, so part of any gain is simply more training time.
- Trees treat each pixel as an independent feature, so XGBoost is expected to stay below the CNNs.
"""


def _cont(r):
    """Note for a run continued from the Phase 8 model: where it started and how the time splits."""
    if not r.get("init_rounds"):
        return ""
    return (f" (continued from Phase 8 round {r['init_rounds']}: {r['init_fit_time_s']:.0f} s in Phase 8 "
            f"+ {r['extra_fit_time_s']:.0f} s in Phase 8b)")


def _append_records(p8, ext, rs, s, wb):
    """Append the results to the project notes (append only)."""
    lines = ["", f"### Phase 8b results (auto-appended by `src/xgb_ext.py final`, {datetime.now():%Y-%m-%d %H:%M})"]
    if s:
        lines.append(f"- `{WIDE}`: {len(s.trials)} trials {_states(s)}, search {s.user_attrs.get('search_wall_s', 0):.0f} s. "
                     "Best converged: " + (f"trial {wb['trial']}, dev {100 * wb['acc']:.2f}, {wb['attrs']['rounds']} rounds, "
                                           f"{wb['params']}." if wb else "none."))
    for r in rs:
        lines.append(f"- 50-class run {r['tag']} ({r['source']}): {r['params']}; {r['rounds_trained']} rounds in total, "
                     f"best round {r['rounds']}, {r['status']}, fit {r['fit_time_s']:.0f} s{_cont(r)}, dev {r['dev_acc']:.2f}.")
    for k in (3, 10, 50):
        e, a = ext[k], p8[k]
        lines.append(f"- Final {k} classes: " + (
            f"{e['note']}, dev {e['dev']:.2f}, test {e['test']:.2f}, {e['rounds']} rounds, fit {e['fit']:.0f} s." if e else
            "no converged run, Phase 8 result stays the only one." if k == 50 else
            f"Phase 8 kept, dev {a['dev_acc']:.2f}, test {a['test_acc']:.2f}."))
    with open(os.path.join(ROOT, "DECISIONS.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    summ = "; ".join(f"{k} cls " + (f"test {p8[k]['test_acc']:.2f} -> {ext[k]['test']:.2f}" if ext[k] else "unchanged")
                     for k in (3, 10, 50))
    with open(os.path.join(ROOT, "PROGRESS.md"), "a") as f:
        f.write(f"- Phase 8b finished {datetime.now():%Y-%m-%d %H:%M} (auto-appended): {summ}. "
                "Details: results/xgb_improvement.md, results/tables/xgb_capped_vs_extended.csv.\n")


# ---------------------------------------------------------------- Phase 8 protection check
def _p8_dump(db):
    """Every row of the three Phase 8 studies, read with SQLite in read-only mode."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = ("xgb_10", "xgb_50", "xgb_3")
    j = "JOIN trials t USING(trial_id) JOIN studies s ON s.study_id = t.study_id WHERE s.study_name IN (?,?,?)"
    out = [sorted(con.execute(q, names).fetchall()) for q in (
        "SELECT s.study_name, t.number, t.state, t.datetime_start, t.datetime_complete FROM trials t "
        "JOIN studies s USING(study_id) WHERE s.study_name IN (?,?,?)",
        f"SELECT s.study_name, t.number, p.param_name, p.param_value FROM trial_params p {j}",
        f"SELECT s.study_name, t.number, v.objective, v.value FROM trial_values v {j}",
        f"SELECT s.study_name, t.number, a.key, a.value_json FROM trial_user_attributes a {j}")]
    con.close()
    return out


def verify():
    """True if every Phase 8 file equals its backup copy, all_results.csv only grew, Phase 8 studies unchanged."""
    bad = []
    for sub, live in (("runs", RUNS), ("models", os.path.join(RES, "models")),
                      ("figures", os.path.join(RES, "figures")), ("logs", os.path.join(ROOT, "logs")),
                      ("optuna", os.path.join(RES, "optuna"))):
        for p in glob.glob(os.path.join(BACKUP, sub, "**", "*"), recursive=True):
            q = os.path.join(live, os.path.relpath(p, os.path.join(BACKUP, sub)))
            if os.path.isfile(p) and not p.endswith("optuna/xgb.db") and not (
                    os.path.exists(q) and filecmp.cmp(p, q, shallow=False)):
                bad.append(os.path.relpath(q, ROOT))
    for f in ("xgb.py", "launch_xgb.sh"):
        if not filecmp.cmp(os.path.join(BACKUP, f), os.path.join(ROOT, "src", f), shallow=False):
            bad.append("src/" + f)
    old = open(os.path.join(BACKUP, "all_results_at_backup.csv"), "rb").read()
    if open(os.path.join(RES, "all_results.csv"), "rb").read(len(old)) != old:
        bad.append("results/all_results.csv (rows before the backup changed)")
    if _p8_dump(os.path.join(BACKUP, "optuna", "xgb.db")) != _p8_dump(P8_DB):
        bad.append("results/optuna/xgb.db (Phase 8 studies changed)")
    print("verify: Phase 8 artefacts unchanged" if not bad else "verify: CHANGED: " + ", ".join(bad))
    return not bad


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run(sys.argv[2])
    elif cmd in ("wide", "final"):
        globals()[cmd]()
    elif cmd == "verify":
        sys.exit(0 if verify() else 1)
    else:
        raise SystemExit(__doc__)

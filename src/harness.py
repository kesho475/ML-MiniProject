"""Shared experiment harness: every run, any model, writes the same outputs.

    with Run("LR", "logreg", 3, "raw", params, hardware="CPU") as r:
        with r.fit_timer():
            clf.fit(...)
        r.finish(y_dev, pred_dev, class_names, y_test=y_test, pred_test=pred_test)

Outputs: results/runs/<id>.json, a row in results/all_results.csv, confusion
matrix (.npy + normalized .png) in results/figures/<id>/, logs/<id>.log.
CNN / transfer learning also get history_csv(), curve_png() and checkpoint_cb().
"""
import csv
import fcntl
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
FIGS = os.path.join(ROOT, "results", "figures")
MODELS = os.path.join(ROOT, "results", "models")
LOGS = os.path.join(ROOT, "logs")
ALL_CSV = os.path.join(ROOT, "results", "all_results.csv")
ACTIVE = os.path.join(RUNS, "active")  # one marker file per running run
COLS = ["run_id", "family", "model", "classes", "input", "params", "fit_time_s", "total_time_s",
        "dev_acc", "test_acc", "hardware", "versions", "seed", "date", "concurrent"]
LIBS = ("numpy", "scikit-learn", "tensorflow", "keras", "xgboost")


def _versions():
    out = {}
    for lib in LIBS:
        try:
            out[lib] = version(lib)
        except Exception:
            pass
    return out


def seed_all(seed):
    """Seed Python, numpy and (if loaded) TensorFlow. GPU non-determinism remains."""
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].random.set_seed(seed)


class _Tee:
    """Write to the console and the run's log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


class Run:
    def __init__(self, family, model, k, variant, params=None, hardware="CPU", seed=0):
        self.family, self.model, self.k, self.variant = family, model, k, variant
        self.params, self.hardware, self.seed = params or {}, hardware, seed
        self.id = f"{family}_{model}_{k}_{variant}_s{seed}".replace(" ", "")
        self.fit_time = 0.0
        self.fig_dir = os.path.join(FIGS, self.id)
        self.ckpt_dir = os.path.join(MODELS, self.id)
        for d in (RUNS, ACTIVE, self.fig_dir, LOGS):
            os.makedirs(d, exist_ok=True)

    def __enter__(self):
        self.t0 = time.time()
        self._log = open(os.path.join(LOGS, self.id + ".log"), "a")
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._log)
        self.concurrent = sorted(f[:-7] for f in os.listdir(ACTIVE))  # runs already going
        open(os.path.join(ACTIVE, self.id + ".active"), "w").close()
        print(f"[{self.id}] start {datetime.now():%F %T} concurrent={self.concurrent}")
        seed_all(self.seed)
        return self

    def __exit__(self, *exc):
        marker = os.path.join(ACTIVE, self.id + ".active")
        if os.path.exists(marker):
            os.remove(marker)
        sys.stdout = self._stdout
        self._log.close()

    @contextmanager
    def fit_timer(self):
        """Time the training call only (fit_time_s). Can be used more than once (e.g. 2 stages)."""
        t = time.time()
        yield
        self.fit_time += time.time() - t

    def finish(self, y_dev, pred_dev, class_names, y_test=None, pred_test=None):
        """Save confusion matrix, JSON and CSV row. Returns the result dict."""
        dev_acc = float(np.mean(y_dev == pred_dev))
        test_acc = None if y_test is None else float(np.mean(y_test == pred_test))
        self._confusion(y_dev, pred_dev, class_names)
        res = dict(run_id=self.id, family=self.family, model=self.model, classes=self.k,
                   input=self.variant, params=self.params, fit_time_s=round(self.fit_time, 2),
                   total_time_s=round(time.time() - self.t0, 2),
                   dev_acc=round(100 * dev_acc, 2),
                   test_acc=None if test_acc is None else round(100 * test_acc, 2),
                   hardware=self.hardware, versions=_versions(), seed=self.seed,
                   date=datetime.now().isoformat(timespec="seconds"), concurrent=self.concurrent)
        with open(os.path.join(RUNS, self.id + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        self._append_csv(res)
        print(f"[{self.id}] dev_acc={res['dev_acc']} test_acc={res['test_acc']} "
              f"fit={res['fit_time_s']}s total={res['total_time_s']}s")
        return res

    def _append_csv(self, res):
        row = {c: json.dumps(res[c]) if isinstance(res[c], (dict, list)) else res[c] for c in COLS}
        with open(ALL_CSV, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # parallel runs share this file
            w = csv.DictWriter(f, COLS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)

    def _confusion(self, y, pred, names):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = len(names)
        cm = np.zeros((k, k), dtype=np.int64)
        np.add.at(cm, (y, pred), 1)
        np.save(os.path.join(self.fig_dir, "confusion_dev.npy"), cm)
        n = min(k, 10)  # like the report's Fig. 8: first 10 classes of a big matrix
        norm = (cm / cm.sum(axis=1, keepdims=True))[:n, :n]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), names[:n], rotation=60, ha="right")
        ax.set_yticks(range(n), names[:n])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{self.id} (dev, row-normalized)")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, "confusion_dev.png"), dpi=120)
        plt.close(fig)

    # --- CNN / transfer learning helpers ---------------------------------
    def history_csv(self, history):
        """Save a Keras History as a per-epoch CSV (train/dev loss and accuracy)."""
        h = history.history if hasattr(history, "history") else history
        keys = list(h)
        with open(os.path.join(self.fig_dir, "history.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch"] + keys)
            for i in range(len(h[keys[0]])):
                w.writerow([i + 1] + [h[k][i] for k in keys])

    def curve_png(self, history):
        """Training curve (loss and accuracy, train vs dev), like the report's Fig. 9."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = history.history if hasattr(history, "history") else history
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, m in zip(axs, ("accuracy", "loss")):
            ax.plot(h[m], label="train")
            ax.plot(h["val_" + m], label="dev")
            ax.set_xlabel("epoch")
            ax.set_title(m)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, "curve.png"), dpi=120)
        plt.close(fig)

    def checkpoint_cb(self):
        """Keras callback: state saved every epoch (more often than the 10 the plan asks);
        a rerun with the same run id resumes fit() from the last saved epoch."""
        import keras
        return keras.callbacks.BackupAndRestore(self.ckpt_dir, delete_checkpoint=False)


def _demo():
    """Dummy run: LR on 3 classes, max_iter=5. Writes JSON, CSV row, confusion PNG and log."""
    from sklearn.linear_model import LogisticRegression
    from data import load
    with Run("LR", "logreg", 3, "raw", {"max_iter": 5}, seed=0) as r:
        xtr, ytr, xdv, ydv, xte, yte, names = load(3, "raw")
        clf = LogisticRegression(max_iter=5, random_state=0)
        with r.fit_timer():
            clf.fit(xtr, ytr)
        res = r.finish(ydv, clf.predict(xdv), names, yte, clf.predict(xte))
    assert os.path.exists(os.path.join(RUNS, r.id + ".json"))
    assert os.path.exists(os.path.join(r.fig_dir, "confusion_dev.png"))
    assert os.path.exists(os.path.join(LOGS, r.id + ".log"))
    assert 30 < res["dev_acc"] <= 100
    print("harness demo OK")


if __name__ == "__main__":
    _demo()

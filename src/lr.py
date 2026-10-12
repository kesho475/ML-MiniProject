"""Phase 5: Logistic Regression (report Table 1). Usage: python src/lr.py [k ...] (default 3 10 50)."""
import sys
import warnings

from sklearn.linear_model import LogisticRegression

from data import load
from harness import Run

# lbfgs is multinomial by default for K>2; sklearn 1.9 no longer takes multi_class.
PARAMS = {"solver": "lbfgs", "max_iter": 100, "random_state": 0}


def main(k):
    with Run("LR", "logreg", k, "raw", PARAMS, seed=0) as r:
        xtr, ytr, xdv, ydv, xte, yte, names = load(k, "raw")
        clf = LogisticRegression(**PARAMS)
        with r.fit_timer(), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ConvergenceWarning expected at 100 iters (log once below)
            clf.fit(xtr, ytr)
        print(f"n_iter={clf.n_iter_} (max 100; convergence warning expected)")
        r.finish(ydv, clf.predict(xdv), names, yte, clf.predict(xte))


if __name__ == "__main__":
    for k in map(int, sys.argv[1:] or [3, 10, 50]):
        main(k)

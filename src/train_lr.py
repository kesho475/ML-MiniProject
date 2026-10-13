"""Logistic regression on raw pixels.

    python src/train_lr.py --classes 3 10 50
"""
import argparse
import warnings

from sklearn.linear_model import LogisticRegression

from common import Run, add_out_arg, set_out
from data import load


def main(k, max_iter=100):
    # lbfgs is multinomial for more than 2 classes.
    params = {"solver": "lbfgs", "max_iter": max_iter, "random_state": 0}
    with Run("LR", "logreg", k, "raw", params, seed=0) as r:
        xtr, ytr, xdv, ydv, xte, yte, names = load(k, "raw")
        clf = LogisticRegression(**params)
        with r.fit_timer(), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # stops at max_iter before converging, as in the reference
            clf.fit(xtr, ytr)
        print(f"n_iter={clf.n_iter_}")
        r.finish(ydv, clf.predict(xdv), names, yte, clf.predict(xte))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", type=int, nargs="+", choices=[3, 10, 50], default=[3, 10, 50])
    ap.add_argument("--max-iter", type=int, default=100)
    add_out_arg(ap)
    a = ap.parse_args()
    set_out(a.out)
    for k in a.classes:
        main(k, a.max_iter)

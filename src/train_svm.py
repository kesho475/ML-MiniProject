"""SVM on inputs scaled to [-1, 1]; records validation accuracy only.

    python src/train_svm.py --kernel rbf --classes 3
    python src/train_svm.py --kernel rbf --classes 3 --70k     # 70,000 drawings per class (run data.py split --70k first)
"""
import argparse
import warnings

from sklearn.svm import SVC

from common import Run, add_out_arg, set_out
from data import load

KERNELS = {  # everything else default (C=1)
    "linear": {},
    "poly": {"degree": 5, "coef0": 1},
    "rbf": {"gamma": 1},
    "sigmoid": {"coef0": 1},
}


def main(kernel, k, n70k=False, max_iter=500):
    # max_iter=500 stops far from convergence; kept to match the reference.
    params = {"kernel": kernel, "max_iter": max_iter, **KERNELS[kernel]}
    with Run("svm70k" if n70k else "SVM", kernel, k, "scaled", params, seed=0) as r:
        xtr, ytr, xdv, ydv, _, _, names = load(k, "scaled", 70_000 if n70k else 20_000)
        clf = SVC(**params)
        with r.fit_timer(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(xtr, ytr)
        r.finish(ydv, clf.predict(xdv), names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", choices=list(KERNELS), required=True)
    ap.add_argument("--classes", type=int, choices=[3, 10], required=True)
    ap.add_argument("--70k", dest="n70k", action="store_true", help="use 70,000 drawings per class")
    ap.add_argument("--max-iter", type=int, default=500)
    add_out_arg(ap)
    a = ap.parse_args()
    set_out(a.out)
    main(a.kernel, a.classes, a.n70k, a.max_iter)

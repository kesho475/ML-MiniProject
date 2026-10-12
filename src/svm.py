"""Phase 6A: SVM (report Table 2). Usage: python src/svm.py <linear|poly|rbf|sigmoid> <k> [--smoke] [--70k]

SVC with max_iter=500 (Q9: far from converged, kept on purpose); scaled input; dev accuracy only.
--smoke: max_iter=5, run id gets a "-smoke" model name so it never counts as a real result.
--70k: first 70,000 drawings per class (Q4 check); family "svm70k" so it never replaces the main SVM rows.
"""
import sys
import warnings

from sklearn.svm import SVC

from data import load
from harness import Run

KERNELS = {  # spec: poly degree=5 coef0=1, rbf gamma=1, sigmoid coef0=1, everything else default (C=1)
    "linear": {},
    "poly": {"degree": 5, "coef0": 1},
    "rbf": {"gamma": 1},
    "sigmoid": {"coef0": 1},
}


def main(kernel, k, smoke=False, n70k=False):
    params = {"kernel": kernel, "max_iter": 5 if smoke else 500, **KERNELS[kernel]}
    with Run("svm70k" if n70k else "SVM", kernel + ("-smoke" if smoke else ""), k, "scaled", params, seed=0) as r:
        xtr, ytr, xdv, ydv, _, _, names = load(k, "scaled", 70_000 if n70k else 20_000)
        clf = SVC(**params)
        with r.fit_timer(), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ConvergenceWarning expected at 500 iters
            clf.fit(xtr, ytr)
        r.finish(ydv, clf.predict(xdv), names)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), "--smoke" in sys.argv, "--70k" in sys.argv)

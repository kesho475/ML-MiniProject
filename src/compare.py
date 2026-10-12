"""Print Reported vs Ours: joins results/reported.csv with the latest matching run in all_results.csv.

Run: python src/compare.py [family]      (family = LR / SVM / CNN / TL; default all)
     python src/compare.py XGB          (no reported numbers: Phase 8 xgb_capped vs Phase 8b xgb_extended)
     python src/compare.py svm70k       (70k-per-class SVMs next to reported and our 20k SVMs)
Join key: family, model, input, classes. Rows with split=test compare test_acc.
Time: reported total time vs our fit_time_s / total_time_s (TL: reported is s per epoch).
"""
import os
import sys
import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
KEY = ["family", "model", "input", "classes"]


def _ours():
    ours = pd.read_csv(os.path.join(R, "all_results.csv"))
    return ours.sort_values("date").drop_duplicates(KEY, keep="last")  # latest rerun wins


def table(family=None):
    rep = pd.read_csv(os.path.join(R, "reported.csv"))
    ours = _ours()
    df = rep.merge(ours, on=KEY, how="left")
    df["our_acc"] = df.dev_acc.where(df.split == "dev", df.test_acc)
    df["diff"] = (df.our_acc - df.acc).round(2)
    df = df.rename(columns={"acc": "rep_acc", "time_s": "rep_time", "fit_time_s": "our_fit", "total_time_s": "our_total"})
    if family:
        df = df[df.family == family]
    return df[KEY + ["split", "rep_acc", "our_acc", "diff", "rep_time", "our_fit", "our_total"]]


def ours_only(family):
    """Our latest runs of a family that has no reported numbers (e.g. the XGB extension)."""
    o = _ours()
    o = o[(o.family == family) & ~o.model.str.endswith("-smoke")]
    return o.sort_values("classes")[KEY + ["dev_acc", "test_acc", "fit_time_s", "total_time_s"]]


def xgb():
    """XGB extension: Phase 8 (model xgb = label xgb_capped) next to Phase 8b (model xgb_extended).
    Empty ext_ columns: Phase 8 kept for that class size, or Phase 8b not finished yet."""
    o = ours_only("XGB").set_index("classes")
    cols = ["dev_acc", "test_acc", "fit_time_s"]
    cap = o[o.model == "xgb"][cols].add_prefix("capped_")
    ext = o[o.model == "xgb_extended"][cols].add_prefix("ext_")
    return cap.join(ext, how="left").reset_index()


def svm70k():
    """Reported SVM (Table 2) vs our 20k-per-class SVMs vs our 70k-per-class reruns (Q4 check)."""
    t = table("SVM").drop(columns=["family", "split", "diff", "our_total"])
    t = t.rename(columns={"our_acc": "acc_20k", "our_fit": "fit_20k"})
    s = ours_only("svm70k").rename(columns={"dev_acc": "acc_70k", "fit_time_s": "fit_70k"})
    return t.merge(s[["model", "input", "classes", "acc_70k", "fit_70k"]],
                   on=["model", "input", "classes"], how="left")


if __name__ == "__main__":
    fam = sys.argv[1] if len(sys.argv) > 1 else None
    out = svm70k() if fam == "svm70k" else xgb() if fam == "XGB" else table(fam)
    print(out.to_string(index=False))

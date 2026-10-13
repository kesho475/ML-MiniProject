"""Every results table and figure, rebuilt from the saved results with one command.

Run: python src/report.py
Reads   results/all_results.csv (our runs) and results/reported.csv (the reference's numbers).
Tables  -> results/tables/*.csv: presentation-ready (1 decimal, whole seconds, "—" plus a reason, no blank cells).
Figures -> results/figures/*.png: new charts are drawn there; figures written by the model runs (confusion
           matrices, training curves, XGBoost curves) are copied in from their per-run folders when present.
Smoke-test runs (model names ending in -smoke) are left out of everything.
"""
import json
import os
import shutil
from decimal import ROUND_HALF_UP, Decimal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import compare
import data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
DASH = "—"

# Colours: default dataviz palette, light mode. Text stays in ink colours, never series colours.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, REPORTED = "#2a78d6", "#eb6834"
FAMILY_STYLE = {  # colour + marker shape, so family is never shown by colour alone
    "LR": ("#2a78d6", "o"), "SVM": ("#eb6834", "s"), "CNN": ("#1baf7a", "^"),
    "TL": ("#6250d6", "D"), "XGB": ("#e34948", "P"),
}
FAMILY_NAME = {"LR": "Logistic regression", "SVM": "SVM", "CNN": "CNN",
               "TL": "Transfer learning", "XGB": "XGBoost"}
MODEL_NAME = {"logreg": "Logistic regression", "linear": "Linear", "rbf": "RBF", "poly": "Polynomial",
              "sigmoid": "Sigmoid", "vgg19": "VGG19", "mobilenet": "MobileNet", "resnet50": "ResNet50",
              "inception": "InceptionV3", "xgb": "XGBoost", "xgb_extended": "XGBoost"}
MODEL_ORDER = ["logreg", "linear", "rbf", "poly", "sigmoid", "V1", "V2", "V3", "V4",
               "vgg19", "mobilenet", "resnet50", "inception", "xgb", "xgb_extended"]
TL_BATCHNORM = {"vgg19": False, "mobilenet": True, "resnet50": True, "inception": True}

# Plain-English reason for every accuracy gap above 2 points.
SVM_DATA = ("Their SVM runs used about 70,000 drawings per class; with fewer drawings the same "
            "500-iteration cap gets closer to convergence")
V2_LABEL = "The original code ran V1 under the V2 label (a bug in their script)"
BATCHNORM = "Frozen BatchNorm behaved differently in 2018 Keras, which lowered their scores"
UNEXPLAINED = "Unexplained; likely seed or library-version differences"
CAUSES = {
    ("SVM", "linear", 3): SVM_DATA, ("SVM", "rbf", 3): SVM_DATA, ("SVM", "rbf", 10): SVM_DATA,
    ("SVM", "poly", 3): SVM_DATA, ("SVM", "poly", 10): SVM_DATA,
    ("CNN", "V2", 50): V2_LABEL, ("CNN", "V3", 10): UNEXPLAINED, ("CNN", "V3", 50): UNEXPLAINED,
    ("TL", "mobilenet", 3): BATCHNORM, ("TL", "resnet50", 3): BATCHNORM, ("TL", "inception", 3): BATCHNORM,
}
TL_TIME = ("Reported time is seconds per epoch for the full network; ours is feature extraction "
           "plus head training in total")

REPRO_COLS = ["Model", "Input", "Classes", "Reported accuracy (%)", "Our accuracy (%)", "Difference (points)",
              "Reported training time (s)", "Our training time (s)", "Explanation"]


def f1(x):
    """One decimal, half-up from the CSV value (95.35 -> 95.4)."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def s0(x):
    """Whole seconds, half-up."""
    return str(int(Decimal(repr(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))


def no_smoke(df, col="model"):
    df = df[~df[col].astype(str).str.endswith("-smoke")]
    assert not df[col].astype(str).str.contains("smoke").any()
    return df


def _write(df, name):
    assert not df.isna().any().any() and not (df.astype(str) == "").any().any(), f"blank cell in {name}"
    df.to_csv(os.path.join(TAB, name), index=False)


# ---------- internal frames (raw numbers, used by the charts) ----------
def ours_final():
    """Our final runs: LR, SVM (20k), CNN, TL and the final XGBoost per class size (the extended run where it
    exists, else the first tuned run). time_s = training time; for TL it adds the one-off feature extraction."""
    o = no_smoke(compare._ours())
    o = o[o.family.isin(list(FAMILY_STYLE))]
    xgb = o[o.family == "XGB"]
    final = xgb[xgb.model == "xgb_extended"]
    final = pd.concat([final, xgb[(xgb.model == "xgb") & ~xgb.classes.isin(final.classes)]])
    o = pd.concat([o[o.family != "XGB"], final])
    extract = o.params.map(lambda p: json.loads(p).get("extract_time_s", 0.0))
    o = o.assign(time_s=o.fit_time_s + extract)
    rank = {c: {v: i for i, v in enumerate(vals)} for c, vals in [
        ("family", list(FAMILY_STYLE)), ("model", list(MODEL_ORDER)), ("input", ["raw", "binarized", "scaled", "resized"])]}
    return o.sort_values(["family", "classes", "model", "input"], key=lambda c: c.map(rank[c.name]) if c.name in rank else c)


# ---------- presentation tables ----------
def reproduction_tables(cm):
    t = compare.table()
    tl_time = cm[cm.family == "TL"].set_index("model").time_s
    for fam in ["LR", "SVM", "CNN", "TL"]:
        d = t[t.family == fam].sort_values("classes", kind="stable")
        rows = []
        for r in d.itertuples():
            gap = abs(r.diff)
            why = ("Reproduced: within 1 point" if gap <= 1 else "Reproduced: within 2 points" if gap <= 2
                   else CAUSES[(r.family, r.model, r.classes)])
            if r.split == "test":
                why = "Test-set result. " + why + ". Reported training time: not given for the test set"
            if fam == "TL":
                why += ". " + TL_TIME
            our_time = tl_time[r.model] if fam == "TL" else r.our_fit
            rows.append([
                MODEL_NAME.get(r.model, r.model) + (" (test set)" if r.split == "test" else ""),
                r.input, r.classes, f1(r.rep_acc), f1(r.our_acc), f1(r.diff),
                DASH if pd.isna(r.rep_time) else s0(r.rep_time), s0(our_time), why,
            ])
        _write(pd.DataFrame(rows, columns=REPRO_COLS), f"reproduction_{fam}.csv")
    return t


def svm_table():
    s = compare.svm70k().sort_values("classes", kind="stable")
    out = pd.DataFrame({
        "Kernel": s.model.map(MODEL_NAME), "Classes": s.classes,
        "Reported accuracy (%)": s.rep_acc.map(f1), "Ours at 20k per class (%)": s.acc_20k.map(f1),
        "Ours at 70k per class (%)": s.acc_70k.map(f1),
        "Our training time at 20k (s)": s.fit_20k.map(s0), "Our training time at 70k (s)": s.fit_70k.map(s0),
    })
    _write(out, "svm_reported_20k_70k.csv")


def cross_model_table(cm):
    def why(r):
        if pd.isna(r.test_acc):
            return "Test accuracy not measured (validation only)"
        if r.family == "TL":
            return "Training time includes the one-off feature extraction"
        if r.model == "xgb":
            return "First tuned model kept (it had already converged)"
        if r.model == "xgb_extended":
            return "Final tuned model from the extended run"
        return DASH
    out = pd.DataFrame({
        "Model family": cm.family.map(FAMILY_NAME), "Model": cm.model.map(lambda m: MODEL_NAME.get(m, m)),
        "Input": cm.input, "Classes": cm.classes,
        "Validation accuracy (%)": cm.dev_acc.map(f1),
        "Test accuracy (%)": cm.test_acc.map(lambda v: DASH if pd.isna(v) else f1(v)),
        "Training time (s)": cm.time_s.map(s0), "Explanation": [why(r) for r in cm.itertuples()],
    })
    _write(out, "cross_model_ours.csv")


def xgb_table():
    o = no_smoke(compare._ours())
    o = o[o.family == "XGB"].set_index(["model", "classes"])
    rows = []
    for k in (3, 10, 50):
        cap = o.loc[("xgb", k)]
        ext = o.loc[("xgb_extended", k)] if ("xgb_extended", k) in o.index else cap  # 3 classes: first model kept
        pc, pe = json.loads(cap.params), json.loads(ext.params)
        conv_c = "no" if pc.get("n_capped") else "yes"  # the first 50-class runs were stopped by a time cap
        conv_e = ("yes" if pe.get("converged") else "no") if "converged" in pe else conv_c
        gain = (Decimal(repr(float(ext.test_acc))) - Decimal(repr(float(cap.test_acc))))
        rows.append([k, f1(cap.dev_acc), f1(cap.test_acc), f1(ext.dev_acc), f1(ext.test_acc), f1(gain),
                     f"{pc['rounds']} → {pe['rounds']}", f"{conv_c} → {conv_e}",
                     f"{s0(cap.fit_time_s)} → {s0(ext.fit_time_s)}"])
    _write(pd.DataFrame(rows, columns=[
        "Classes", "Capped validation (%)", "Capped test (%)", "Extended validation (%)", "Extended test (%)",
        "Test gain (points)", "Rounds (capped → extended)", "Converged", "Training time (s)"]),
        "xgb_capped_vs_extended.csv")


# ---------- charts ----------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def acc_vs_time(cm):
    """One small multiple per class size so every panel has few enough points to read."""
    sizes = sorted(cm.classes.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 4), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for ax, k in zip(axes, sizes):
        _style(ax)
        ax.tick_params(labelsize=11)  # read from a slide, so larger than the other charts
        for fam, (col, mk) in FAMILY_STYLE.items():
            d = cm[(cm.classes == k) & (cm.family == fam)]
            if len(d):
                ax.scatter(d.time_s, d.dev_acc, s=80, color=col, marker=mk, edgecolors=SURFACE,
                           linewidths=1.5, label=FAMILY_NAME[fam], zorder=3)
        best = cm[cm.classes == k].nlargest(1, "dev_acc").iloc[0]
        ax.annotate(f"{MODEL_NAME.get(best.model, best.model)} {f1(best.dev_acc)}", (best.time_s, best.dev_acc),
                    xytext=(6, -12), textcoords="offset points", fontsize=11, color=INK)
        ax.set_xscale("log")
        ax.set_title(f"{k} classes", fontsize=14, color=INK, loc="left")
        ax.set_xlabel("training time, s (log scale)", fontsize=12, color=INK2)
    axes[0].set_ylabel("validation accuracy, %", fontsize=12, color=INK2)
    seen = {}
    for ax in axes:  # not every family exists at every class size (transfer learning: 3 classes only)
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    l = [FAMILY_NAME[f] for f in FAMILY_STYLE]
    h = [seen[x] for x in l]
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=12, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.1))
    fig.text(0.0, -0.09, "Our runs only, one point per model, input and class size. Transfer learning time = feature "
             "extraction + head training; XGBoost = final tuned model.", fontsize=10, color=INK2)
    _save(fig, "accuracy_vs_training_time.png")


def _bars(ax, d, labels, title):
    _style(ax)
    ax.grid(axis="x", visible=False)
    x = range(len(d))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], d.rep_acc, w, color=REPORTED, label="Reported")
    ax.bar([i + w / 2 + 0.01 for i in x], d.our_acc, w, color=OURS, label="Ours")
    ax.set_xticks(list(x), labels, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def grouped_bars(rep):
    d = rep[rep.split == "dev"]
    lab = lambda r, model=True: ((f"{MODEL_NAME.get(r.model, r.model)}\n" if model else "")
                                 + (f"{r.input}\n" if r.family == "CNN" else "") + f"{r.classes} cls")
    fig = plt.figure(figsize=(14, 7.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[3, 8, 4], hspace=0.55, wspace=0.18)
    panels = [(gs[0, 0], "LR", False), (gs[0, 1], "SVM", True), (gs[0, 2], "TL", True), (gs[1, :], "CNN", True)]
    for spec, fam, show_model in panels:
        ax = fig.add_subplot(spec)
        f = d[d.family == fam]
        names = [lab(r, show_model) if fam != "TL" else MODEL_NAME[r.model] for r in f.itertuples()]
        _bars(ax, f, names, FAMILY_NAME[fam])
        if fam == "LR":
            ax.set_ylabel("validation accuracy, %", fontsize=8, color=INK2)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, "reported_vs_ours_accuracy.png")


def tl_batchnorm(rep):
    d = rep[rep.family == "TL"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)
    names = [f"{MODEL_NAME[m]}\n" + ("BatchNorm" if TL_BATCHNORM[m] else "no BatchNorm") for m in d.model]
    _bars(ax, d, names, "Transfer learning, 3 classes: reported vs ours (validation)")
    for i, r in enumerate(d.itertuples()):
        if TL_BATCHNORM[r.model]:  # texture marks the BatchNorm networks, not colour alone
            ax.bar(i + 0.2, r.our_acc, 0.38, color="none", edgecolor=SURFACE, hatch="///", linewidth=0)
        ax.text(i + 0.2, r.our_acc + 1.5, f1(r.our_acc), ha="center", fontsize=7, color=INK)
        ax.text(i - 0.2, r.rep_acc + 1.5, f1(r.rep_acc), ha="center", fontsize=7, color=INK2)
    ax.set_ylabel("validation accuracy, %", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.0, -0.06, "Hatched = network with BatchNorm. Frozen BatchNorm behaved differently in 2018 Keras;\n"
             "VGG19 has no BatchNorm and is the only one that matches.", fontsize=7, color=INK2)
    _save(fig, "transfer_learning_batchnorm.png")


def sample_grid(per_class=5):
    """The 10-class set, first `per_class` training drawings of each class, one row per class."""
    out = os.path.join(FIG, "sample_drawings_10_classes.png")
    if not os.path.exists(data._path(10)):  # data not downloaded: keep the committed figure
        assert os.path.exists(out), "no data and no saved sample grid"
        return
    xtr, ytr, *_, names = data.load(10, "uint8")
    fig, axes = plt.subplots(len(names), per_class, figsize=(per_class * 0.9 + 1.6, len(names) * 0.9),
                             facecolor="white")
    for c, name in enumerate(names):
        for j, img in enumerate(xtr[ytr == c][:per_class]):
            ax = axes[c, j]
            ax.imshow(img.reshape(28, 28), cmap="gray_r", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
        axes[c, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10, color=INK)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)


COPIES = {  # final name in results/figures -> figure written by a model run (relative to results/figures)
    "confusion_logistic_regression_10_classes.png": "LR_logreg_10_raw_s0/confusion_dev.png",
    "confusion_cnn_v2_50_classes.png": "CNN_V2_50_raw_s7/confusion_dev.png",
    "confusion_xgboost_50_classes.png": "XGB_xgb_extended_50_raw_s0/confusion_dev.png",
    "cnn_v2_50_classes_training_curve.png": "CNN_V2_50_raw_s7/curve.png",
    "xgboost_50_classes_capped_vs_converged.png": "xgb_extended/xgb50_capped_vs_converged_curve.png",
    "xgboost_capped_vs_extended_test.png": "xgb_extended/xgb_capped_vs_extended_test.png",
}


def copy_run_figures():
    """Per-run folders exist only where the models were trained; elsewhere the committed copies are kept."""
    for name, src in COPIES.items():
        src, dst = os.path.join(FIG, src), os.path.join(FIG, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        assert os.path.isfile(dst), f"missing figure: {name}"


def main():
    os.makedirs(TAB, exist_ok=True)
    cm = ours_final()
    rep = reproduction_tables(cm)
    svm_table()
    cross_model_table(cm)
    xgb_table()
    acc_vs_time(cm)
    grouped_bars(rep)
    tl_batchnorm(rep)
    sample_grid()
    copy_run_figures()
    for f in sorted(os.listdir(TAB)):
        assert "smoke" not in open(os.path.join(TAB, f)).read(), f"smoke row in {f}"
    print("tables:", *sorted(os.listdir(TAB)), sep="\n  ")
    print("figures:", *sorted(f for f in os.listdir(FIG) if f.endswith(".png")), sep="\n  ")


if __name__ == "__main__":
    main()

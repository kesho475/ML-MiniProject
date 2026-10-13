"""Phase 9: every results table and figure, rebuilt from the saved results with one command.

Run: python src/report.py
Tables  -> results/tables/   (CSV)
Figures -> results/figures/phase9/  New charts are drawn here; figures that already exist
           (confusion matrices, training curve, Optuna plots, XGBoost before/after, sample grids)
           are copied in, so the folder is self-contained (the originals stay where they are).
Smoke-test runs (model names ending in -smoke) are left out of every table and chart.
"""
import json
import os
from decimal import ROUND_HALF_UP, Decimal
import shutil

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
OUT = os.path.join(FIG, "phase9")

# Colours: default dataviz palette, light mode. Text stays in ink colours, never series colours.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, REPORTED = "#2a78d6", "#eb6834"
FAMILY_STYLE = {  # colour + marker shape, so family is never shown by colour alone
    "LR": ("#2a78d6", "o"), "SVM": ("#eb6834", "s"), "CNN": ("#1baf7a", "^"),
    "TL": ("#6250d6", "D"), "XGB": ("#e34948", "P"),
}
FAMILY_NAME = {"LR": "Logistic regression", "SVM": "SVM", "CNN": "CNN",
               "TL": "Transfer learning", "XGB": "XGBoost (final)"}
TL_NAME = {"vgg19": "VGG19", "mobilenet": "MobileNet", "resnet50": "ResNet50", "inception": "InceptionV3"}
TL_BATCHNORM = {"vgg19": False, "mobilenet": True, "resnet50": True, "inception": True}

# Likely cause of every accuracy gap above 2 points (DECISIONS.md, PLAN.md Q1-Q9).
Q49 = "Q4 + Q9: reported runs used ~70k drawings per class with max_iter=500; less data gets closer to convergence (svm70k rerun supports this)"
Q1 = "Q1: the reported V2 was probably V1 (original cnn.py overwrote V2-V4 with V1); our V1 matches the reported V2"
Q7 = "Q7: frozen BatchNorm runs in inference mode in TF2, in training mode in 2018 Keras"
V3 = "Not established: Q1 does not fit (reported V3 differs from reported V1); one seed, Keras 3 vs 2018 Keras, Q3 unscaled input"
CAUSES = {
    ("SVM", "linear", 3): Q49, ("SVM", "rbf", 3): Q49, ("SVM", "rbf", 10): Q49,
    ("SVM", "poly", 3): Q49, ("SVM", "poly", 10): Q49,
    ("CNN", "V2", 50): Q1, ("CNN", "V3", 10): V3, ("CNN", "V3", 50): V3,
    ("TL", "mobilenet", 3): Q7, ("TL", "resnet50", 3): Q7, ("TL", "inception", 3): Q7,
}
TIME_NOTE = {
    "LR": "fit only", "SVM": "fit only", "CNN": "fit only (reported time starts before data loading, Q5)",
    "TL": "reported: s per epoch, full network; ours: head-only s in total, see cross-model table",
}


def f1(x):
    """One decimal, half-up from the CSV value (95.35 -> 95.4), same as the slides."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def no_smoke(df, col="model"):
    df = df[~df[col].astype(str).str.endswith("-smoke")]
    assert not df[col].astype(str).str.contains("smoke").any()
    return df


def reproduction_tables():
    t = compare.table()
    t["likely_cause"] = [CAUSES.get((f, m, c), "") if abs(d) > 2 else ""
                         for f, m, c, d in zip(t.family, t.model, t.classes, t["diff"])]
    assert t[(t["diff"].abs() > 2) & (t.likely_cause == "")].empty, "gap above 2 points without a cause"
    t["time_note"] = t.family.map(TIME_NOTE)
    for fam in ["LR", "SVM", "CNN", "TL"]:
        t[t.family == fam].to_csv(os.path.join(TAB, f"reproduction_{fam}.csv"), index=False)
    return t


def svm_table():
    s = compare.svm70k()
    s["diff_20k"] = (s.acc_20k - s.rep_acc).round(2)
    s["diff_70k"] = (s.acc_70k - s.rep_acc).round(2)
    s.to_csv(os.path.join(TAB, "svm_reported_20k_70k.csv"), index=False)


def cross_model_table():
    """Our runs only, same split and hardware. XGBoost = final model per class size
    (Phase 8B where it replaced Phase 8, else Phase 8). svm70k is a check, so it stays in its own table."""
    o = no_smoke(compare._ours())
    o = o[o.family.isin(["LR", "SVM", "CNN", "TL", "XGB"])]
    xgb = o[o.family == "XGB"]
    final = xgb[xgb.model == "xgb_extended"]
    final = pd.concat([final, xgb[(xgb.model == "xgb") & ~xgb.classes.isin(final.classes)]])
    o = pd.concat([o[o.family != "XGB"], final])
    params = o.params.map(json.loads)
    extract = params.map(lambda p: p.get("extract_time_s", 0.0))
    o = o.assign(
        xgb_source=o.model.map({"xgb": "Phase 8 (kept)", "xgb_extended": "Phase 8B"}).fillna(""),
        extract_time_s=extract.where(o.family == "TL"),
        time_s=(o.fit_time_s + extract).round(2),  # TL: one-off feature extraction + head training
    )
    # Phase 8B rows' total_time_s covers only the final test scoring (the search ran in xgb_ext.py), so drop it
    o.loc[o.model == "xgb_extended", "total_time_s"] = float("nan")
    order = {f: i for i, f in enumerate(FAMILY_STYLE)}
    o = o.sort_values(["family", "classes", "model", "input"], key=lambda c: c.map(order) if c.name == "family" else c)
    cols = ["family", "model", "input", "classes", "dev_acc", "test_acc", "fit_time_s", "extract_time_s",
            "time_s", "total_time_s", "xgb_source", "run_id"]
    o[cols].to_csv(os.path.join(TAB, "cross_model_ours.csv"), index=False)
    return o[cols]


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
    fig.savefig(os.path.join(OUT, name), dpi=160, facecolor=SURFACE, bbox_inches="tight")
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
        ax.annotate(f"{best.model} {f1(best.dev_acc)}", (best.time_s, best.dev_acc), xytext=(6, -12),
                    textcoords="offset points", fontsize=11, color=INK)
        ax.set_xscale("log")
        ax.set_title(f"{k} classes", fontsize=14, color=INK, loc="left")
        ax.set_xlabel("training time, s (log scale)", fontsize=12, color=INK2)
    axes[0].set_ylabel("dev accuracy, %", fontsize=12, color=INK2)
    seen = {}
    for ax in axes:  # not every family exists at every class size (TL: 3 classes only)
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    l = [FAMILY_NAME[f] for f in FAMILY_STYLE]
    h = [seen[x] for x in l]
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=12, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.1))
    fig.text(0.0, -0.09, "Our runs only, one point per model, input and class size. TL time = feature extraction "
             "+ head training; XGBoost = final model (Phase 8B at 10/50 classes).", fontsize=10, color=INK2)
    _save(fig, "accuracy_vs_time.png")


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
    lab = lambda r, model=True: (f"{r.model}\n" if model else "") + (f"{r.input}\n" if r.family == "CNN" else "") + f"{r.classes} cls"
    fig = plt.figure(figsize=(14, 7.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[3, 8, 4], hspace=0.55, wspace=0.18)
    panels = [(gs[0, 0], "LR", False), (gs[0, 1], "SVM", True), (gs[0, 2], "TL", True), (gs[1, :], "CNN", True)]
    for spec, fam, show_model in panels:
        ax = fig.add_subplot(spec)
        f = d[d.family == fam]
        names = [lab(r, show_model) if fam != "TL" else TL_NAME[r.model] for r in f.itertuples()]
        _bars(ax, f, names, FAMILY_NAME[fam])
        if fam == "LR":
            ax.set_ylabel("dev accuracy, %", fontsize=8, color=INK2)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, "reported_vs_ours_bars.png")


def tl_batchnorm(rep):
    d = rep[rep.family == "TL"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)
    names = [f"{TL_NAME[m]}\n" + ("BatchNorm" if TL_BATCHNORM[m] else "no BatchNorm") for m in d.model]
    _bars(ax, d, names, "Transfer learning, 3 classes: reported vs ours (dev)")
    for i, r in enumerate(d.itertuples()):
        if TL_BATCHNORM[r.model]:  # texture marks the BatchNorm networks, not colour alone
            ax.bar(i + 0.2, r.our_acc, 0.38, color="none", edgecolor=SURFACE, hatch="///", linewidth=0)
        ax.text(i + 0.2, r.our_acc + 1.5, f1(r.our_acc), ha="center", fontsize=7, color=INK)
        ax.text(i - 0.2, r.rep_acc + 1.5, f1(r.rep_acc), ha="center", fontsize=7, color=INK2)
    ax.set_ylabel("dev accuracy, %", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.0, -0.06, "Hatched = network with BatchNorm. Q7: frozen BatchNorm runs in inference mode in TF2 "
             "but used batch statistics in 2018 Keras;\nVGG19 has no BatchNorm and is the only one that matches.",
             fontsize=7, color=INK2)
    _save(fig, "tl_batchnorm.png")


COPIES = {  # name in phase9/ -> existing figure (relative to results/figures)
    "confusion_LR_10.png": "LR_logreg_10_raw_s0/confusion_dev.png",
    "confusion_CNN_V2_50_raw.png": "CNN_V2_50_raw_s7/confusion_dev.png",
    "confusion_XGB_final_50.png": "XGB_xgb_extended_50_raw_s0/confusion_dev.png",
    "curve_CNN_V2_50_raw.png": "CNN_V2_50_raw_s7/curve.png",
    "optuna_history_xgb_3.png": "XGB_xgb_3_raw_s0/optuna_history.png",
    "optuna_importance_xgb_3.png": "XGB_xgb_3_raw_s0/optuna_importance.png",
    "optuna_history_xgb_10.png": "XGB_xgb_10_raw_s0/optuna_history.png",
    "optuna_importance_xgb_10.png": "XGB_xgb_10_raw_s0/optuna_importance.png",
    "optuna_history_xgb_50.png": "XGB_xgb_50_raw_s0/optuna_history.png",
    "optuna_importance_xgb_50.png": "XGB_xgb_50_raw_s0/optuna_importance.png",
    "optuna_history_xgb10_wide.png": "xgb_extended/xgb10_wide/optuna_history.png",
    "optuna_importance_xgb10_wide.png": "xgb_extended/xgb10_wide/optuna_importance.png",
    "optuna_xgb10_both_studies.png": "xgb_extended/xgb10_both_studies_history.png",
    "xgb_capped_vs_extended_test.png": "xgb_extended/xgb_capped_vs_extended_test.png",
    "xgb50_capped_vs_converged_curve.png": "xgb_extended/xgb50_capped_vs_converged_curve.png",
    "samples": "samples",
}


def copy_existing():
    for name, src in COPIES.items():
        src = os.path.join(FIG, src)
        assert os.path.exists(src), f"missing figure: {src}"
        dst = os.path.join(OUT, name)
        if os.path.islink(dst):  # older runs linked; copying onto a link would write through it
            os.remove(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def sample_grid(per_class=5):
    """One slide image: the 10-class set, first `per_class` training drawings of each class, one row per class."""
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
    fig.savefig(os.path.join(OUT, f"samples_10x{per_class}.png"), dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    rep = reproduction_tables()
    svm_table()
    cm = cross_model_table()
    acc_vs_time(cm)
    grouped_bars(rep)
    tl_batchnorm(rep)
    copy_existing()
    sample_grid()
    assert not [f for f in os.listdir(OUT) if os.path.islink(os.path.join(OUT, f))], "symlink left in phase9/"
    for f in sorted(os.listdir(TAB)):
        assert "smoke" not in open(os.path.join(TAB, f)).read(), f"smoke row in {f}"
    print("tables:", *sorted(os.listdir(TAB)), sep="\n  ")
    print("figures/phase9:", *sorted(os.listdir(OUT)), sep="\n  ")


if __name__ == "__main__":
    main()

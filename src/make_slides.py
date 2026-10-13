"""Build the review deck: presentation/ML_MiniProject.pptx, plus its slide figures in presentation/figures/.

Run: python src/make_slides.py
Every result number is read from results/tables/*.csv at build time (one decimal). The few numbers that come
from the reference's own logs rather than from our results are named constants below, with their source.
Slide figures are drawn here at their final on-slide size, so text stays at least 14 pt.
"""
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
RES_FIG = os.path.join(ROOT, "results", "figures")
OUT_DIR = os.path.join(ROOT, "presentation")
FIG = os.path.join(OUT_DIR, "figures")
OUT = os.path.join(OUT_DIR, "ML_MiniProject.pptx")
CURVES = os.path.join(ROOT, "results", "xgb_extended")  # per-round curves of the 50-class XGBoost run (local only)

# From the reference, not from our results:
TL_EPOCHS = {"VGG19": 20, "MobileNet": 100, "ResNet50": 100, "InceptionV3": 50}  # epochs in the authors' logs
INCEPTION_TRAIN_ACC = "99.99"  # Inception training accuracy, authors' GitHub README log

# Colours: one dark accent; reported = grey, ours = blue, XGBoost = orange; status green / red.
ACCENT = "#1F3A5F"
GREY, BLUE, ORANGE = "#80868B", "#2A78D6", "#E8710A"
GREEN, RED = "#1E8E3E", "#C5221F"
INK, MUTED, ROW_ALT = "#1A1A1A", "#5A5A5A", "#F2F4F7"
FONT = "Arial"
BODY = 18
FOOTER = "Drawing: A New Way To Search — Reproduce & Extend"
MISSING = []


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


# ---------- numbers (all from results/tables) ----------
def f1(x):
    return f"{Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):,}"


def t_read(seconds):
    """Seconds below 100 read best as seconds, longer times as minutes."""
    s = int(seconds)
    return f"{s} s" if s < 100 else f"{f1(s / 60)} min"


def table_csv(name):
    return pd.read_csv(os.path.join(TAB, name), dtype=str)


REPRO = {f: table_csv(f"reproduction_{f}.csv") for f in ["LR", "SVM", "CNN", "TL"]}
SVM70 = table_csv("svm_reported_20k_70k.csv")
CROSS = table_csv("cross_model_ours.csv")
XGB = table_csv("xgb_capped_vs_extended.csv")
FAMILY = {"Logistic regression": "LR", "SVM": "SVM", "CNN": "CNN", "Transfer learning": "TL", "XGBoost": "XGB"}
CROSS["fam"] = CROSS["Model family"].map(FAMILY)
CROSS["acc"] = CROSS["Validation accuracy (%)"].astype(float)
CROSS["time"] = CROSS["Training time (s)"].astype(int)


def rp(fam, model, k, inp=None):
    d = REPRO[fam]
    d = d[(d.Model == model) & (d.Classes == str(k)) & ((d.Input == inp) if inp else True)]
    assert len(d) == 1, (fam, model, k, inp)
    return d.iloc[0]


def svm(kernel, k):
    return SVM70[(SVM70.Kernel == kernel) & (SVM70.Classes == str(k))].iloc[0]


def xgb(k):
    return XGB[XGB.Classes == str(k)].iloc[0]


def best(fam, k):
    d = CROSS[(CROSS.fam == fam) & (CROSS.Classes == str(k))]
    return None if d.empty else d.loc[d.acc.idxmax()]


def matches(row):
    return abs(float(row["Difference (points)"])) <= 2


# ---------- slide figures ----------
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 14,
                     "axes.edgecolor": "#BDBDBD", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def _clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=14, length=0)
    ax.grid(False)


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=200, facecolor="white")
    plt.close(fig)


def fig_samples(w=4.6, h=5.2, per_class=5):
    """The 10-class set, first training drawings of each class, class names at 14 pt."""
    import data
    out = os.path.join(FIG, "sample_drawings_10_classes.png")
    if not os.path.exists(data._path(10)):  # data not downloaded: keep the saved figure
        assert os.path.exists(out), "no data and no saved sample grid"
        return
    xtr, ytr, *_, names = data.load(10, "uint8")
    fig, axes = plt.subplots(len(names), per_class, figsize=(w, h), facecolor="white")
    for c, name in enumerate(names):
        for j, img in enumerate(xtr[ytr == c][:per_class]):
            ax = axes[c, j]
            ax.imshow(img.reshape(28, 28), cmap="gray_r", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#DDDDDD")
        axes[c, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=14, color=INK)
    fig.subplots_adjust(left=0.36, right=0.99, top=0.99, bottom=0.01, wspace=0.06, hspace=0.06)
    _save(fig, "sample_drawings_10_classes.png")


def fig_tl(w=4.1, h=3.9):
    """Horizontal bars, reported (grey) vs ours (blue), BatchNorm networks named as such."""
    d = REPRO["TL"]
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    _clean(ax)
    n = len(d)
    for i, (_, r) in enumerate(d.iterrows()):
        y = n - 1 - i
        rep, ours = float(r["Reported accuracy (%)"]), float(r["Our accuracy (%)"])
        ax.barh(y + 0.19, rep, 0.36, color=GREY)
        ax.barh(y - 0.19, ours, 0.36, color=BLUE)
        first = i == 0
        ax.text(rep + 2, y + 0.19, f"{f1(rep)}" + (" reported" if first else ""), va="center", fontsize=14, color=INK)
        ax.text(ours + 2, y - 0.19, f"{f1(ours)}" + (" ours" if first else ""), va="center", fontsize=14, color=INK)
    names = [f"{m}\n" + ("no BatchNorm" if m == "VGG19" else "BatchNorm") for m in d.Model]
    ax.set_yticks(range(n - 1, -1, -1), names)
    ax.set_xlim(0, 165)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("validation accuracy (%)", fontsize=14)
    fig.tight_layout(pad=0.3)
    _save(fig, "tl_reported_vs_ours.png")


def fig_xgb_curve(w=3.85, h=2.9):
    """Validation accuracy per boosting round on 50 classes: the time-limited first run, then the continued run."""
    first, cont = os.path.join(CURVES, "phase8_50_curve.csv"), os.path.join(CURVES, "runs", "A1", "curve.csv")
    if not (os.path.exists(first) and os.path.exists(cont)):  # curves only exist where the model was trained
        assert os.path.exists(os.path.join(FIG, "xgb_50_learning_curve.png")), "no curve data and no saved figure"
        return
    a, b = pd.read_csv(first), pd.read_csv(cont)
    c = pd.concat([a[["round", "dev_acc"]], b[b["round"] > a["round"].max()][["round", "dev_acc"]]])
    stop = int(a["round"].max())
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    _clean(ax)
    ax.plot(c["round"], c.dev_acc * 100, color=ORANGE, linewidth=2.5)
    ax.axvline(stop, color=GREY, linestyle="--", linewidth=1.5)
    ax.text(stop - 60, 66, "first run\nstopped here\n(time limit)", ha="right", fontsize=14, color=INK)
    plateau = f1(xgb(50)["Extended validation (%)"])
    ax.text(c["round"].max(), 79.2, f"plateau {plateau}%", ha="right", fontsize=14, color=INK)
    ax.set_ylim(60, 81)
    ax.set_yticks([60, 70, 80])
    ax.set_xticks([0, 1000, 2000])
    ax.set_xlabel("trees added", fontsize=14)
    ax.set_ylabel("validation acc. (%)", fontsize=14)
    fig.tight_layout(pad=0.3)
    _save(fig, "xgb_50_learning_curve.png")


def fig_acc_time_50(w=12.1, h=2.55):
    """50 classes: accuracy vs training time (log), every point labelled directly."""
    d = CROSS[CROSS.Classes == "50"].copy()
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    _clean(ax)
    name = lambda r: ("CNN " + r.Model + (" black/white" if r.Input == "binarized" else " raw")
                      if r.fam == "CNN" else r.Model)
    cnn = d[d.fam == "CNN"].sort_values("acc")
    right = set(cnn.index[::2])  # alternate sides so neighbouring labels never share a line
    y_lab = {}
    for side in (right, set(cnn.index) - right):
        prev = -1e9
        for i in cnn.index:
            if i in side:
                prev = max(cnn.acc[i], prev + 6.5)
                y_lab[i] = prev
    for r in d.itertuples():
        col = ORANGE if r.fam == "XGB" else BLUE
        ax.scatter(r.time, r.acc, s=110, color=col, edgecolors="white", linewidths=1.5, zorder=3)
        label = f"{name(r)}  {f1(r.acc)}"
        if r.fam == "CNN":
            rt = r.Index in right
            ax.annotate(label, (r.time, r.acc), xytext=(620 if rt else 230, y_lab[r.Index]), textcoords="data",
                        ha="left" if rt else "right", va="center", fontsize=14, color=INK,
                        arrowprops=dict(arrowstyle="-", color="#BDBDBD", linewidth=1))
        elif r.fam == "XGB":
            ax.annotate(label, (r.time, r.acc), xytext=(-12, -16), textcoords="offset points", ha="right",
                        va="center", fontsize=14, color=INK)
        else:
            ax.annotate(label, (r.time, r.acc), xytext=(12, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=14, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(20, 14000)
    ax.set_xticks([60, 600, 3600], ["1 min", "10 min", "1 h"])
    ax.minorticks_off()
    ax.set_ylim(38, max(y_lab.values()) + 4)
    ax.set_yticks([40, 60, 80])
    ax.set_xlabel("training time (log scale)", fontsize=14)
    ax.set_ylabel("validation acc. (%)", fontsize=14)
    fig.tight_layout(pad=0.3)
    _save(fig, "acc_vs_time_50_classes.png")


# ---------- pptx helpers ----------
def _font(run, size=BODY, bold=False, color=INK):
    run.font.name, run.font.size, run.font.bold, run.font.color.rgb = FONT, Pt(size), bold, rgb(color)


def text(slide, s, x, y, w, h, size=BODY, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        _font(r, size, bold, color)
    return tf


def bullets(slide, items, x, y, w, h, size=BODY, space=12):
    assert len(items) <= 4, "at most 4 bullets"
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        assert len(item.split()) < 15, f"bullet has {len(item.split())} words: {item}"
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.3)))
        pPr.set("indent", str(-Inches(0.3)))
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
        r = p.add_run()
        r.text = item
        _font(r, size)


def title(slide, s, size=28):
    text(slide, s, 0.6, 0.2, 12.1, 1.05, size=size, bold=True, color=ACCENT, anchor=MSO_ANCHOR.MIDDLE)


def takeaway(slide, s):
    tf = slide.shapes.add_textbox(Inches(0.6), Inches(6.42), Inches(12.1), Inches(0.5)).text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = "Takeaway: " + s
    _font(r, BODY, bold=True, color=ACCENT)


def footer(slide, n):
    text(slide, FOOTER, 0.6, 7.05, 10, 0.3, size=10, color=MUTED)
    text(slide, str(n), 11.7, 7.05, 1.0, 0.3, size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def picture(slide, path, x, y, w, h):
    """Fit inside the box, keeping the aspect ratio (centred). Missing -> marked placeholder."""
    if not os.path.isfile(path):
        MISSING.append(os.path.relpath(path, ROOT))
        text(slide, f"[PLACEHOLDER: missing figure {os.path.basename(path)}]", x, y, w, h, align=PP_ALIGN.CENTER)
        return
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, rows, x, y, col_w, row_h=0.42, size=BODY, center_from=1, first_col_bold=False):
    """rows: lists of str or (str, colour, bold). Header row in the accent colour."""
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(col_w)),
                               Inches(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            txt, col, bold = val if isinstance(val, tuple) else (str(val), INK, False)
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(ACCENT if i == 0 else (ROW_ALT if i % 2 == 0 else "#FFFFFF"))
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            for k, line in enumerate(txt.split("\n")):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.CENTER if j >= center_from else PP_ALIGN.LEFT
                r = p.add_run()
                r.text = line
                _font(r, size, bold=(i == 0 or bold or (first_col_bold and j == 0)),
                      color="#FFFFFF" if i == 0 else col)


def box(slide, s, x, y, w, h, size=BODY, title_line=True):
    b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = rgb("#FFFFFF")
    b.line.color.rgb = rgb(ACCENT)
    b.line.width = Pt(1.5)
    b.shadow.inherit = False
    tf = b.text_frame
    tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    for k, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        _font(r, size, bold=(k == 0 and title_line), color=ACCENT if (k == 0 and title_line) else INK)


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(ACCENT)
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle"}))


SECONDS = []


def notes(slide, spoken, reference=""):
    """Spoken notes (70-110 words) with the expected time at ~130 words per minute."""
    words = len(spoken.split())
    assert 70 <= words <= 110, f"speaker notes have {words} words"
    sec = round(words * 60 / 130)
    SECONDS.append(sec)
    body = f"{spoken}\n\n(about {sec} s)"
    if reference:
        body += "\n\n--- Reference, not spoken ---\n" + reference
    slide.notes_slide.notes_text_frame.text = body


# ---------- slides ----------
def build():
    fig_samples()
    fig_tl()
    fig_xgb_curve()
    fig_acc_time_50()

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    slides = []

    def new():
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = rgb("#FFFFFF")
        slides.append(sl)
        return sl

    lr = [rp("LR", "Logistic regression", k) for k in (3, 10, 50)]
    v1, v2 = rp("CNN", "V1", 50, "raw"), rp("CNN", "V2", 50, "raw")
    v3_10, v3_50 = rp("CNN", "V3", 10), rp("CNN", "V3", 50)
    tl = {m: rp("TL", m, 3) for m in ["VGG19", "MobileNet", "ResNet50", "InceptionV3"]}
    rbf3, rbf10 = svm("RBF", 3), svm("RBF", 10)
    c50, x50 = best("CNN", 50), best("XGB", 50)
    speed = x50.time / c50.time

    # 1
    s = new()
    title(s, "Can a computer recognise quick doodles?", size=32)
    text(s, "[Name 1, SRN] · [Name 2, SRN]", 0.6, 1.05, 8, 0.45, color=MUTED)
    bullets(s, [
        "Problem: name the object in a quick hand drawing",
        "Why it matters: drawing could become a search interface",
        "Data: Google QuickDraw, 28×28 grayscale; 3 / 10 / 50 classes, 20,000 each",
        "Split per class: 16,000 train / 2,000 validation / 2,000 test",
    ], 0.6, 1.7, 7.3, 3.6)
    text(s, "Our task: reproduce the paper's results, then add one new model.", 0.6, 5.35, 7.3, 0.8,
         bold=True, color=ACCENT)
    picture(s, os.path.join(FIG, "sample_drawings_10_classes.png"), 8.2, 1.15, 4.6, 5.2)
    takeaway(s, "same simple drawings, but harder as the number of classes grows.")
    notes(s, "Our project asks a simple question: can a computer recognise quick doodles? The reference report "
             "argues that if it can, drawing could become a way to search. The data is Google QuickDraw: tiny "
             "28 by 28 grayscale drawings. We use 3, 10 and 50 classes, with 20,000 drawings per class, split "
             "into 16,000 for training, 2,000 for validation and 2,000 for testing. Our job was to reproduce the paper, then add one new model.")

    # 2
    s = new()
    title(s, "The original team tried four model families, from simple to complex")
    table(s, [
        ["", "Logistic Regression", "SVM", "CNN", "Transfer learning"],
        [("How it works", ACCENT, True), "Draws straight-line boundaries between classes on raw pixels",
         "Finds the widest margin between classes; kernels allow curved boundaries",
         "Small filters slide over the image to detect strokes and shapes",
         "Reuses networks pre-trained on 1.2 M photos (ImageNet); trains only a small new head"],
        [("Why they used it", ACCENT, True), "Fast baseline", "Tests if non-linear boundaries help",
         "Learns spatial patterns", "Tests if photo knowledge helps doodles"],
        [("Key settings", ACCENT, True), "100 iterations", "4 kernels, 500 iterations; 3 and 10 classes only",
         "V1 = 2 conv + 2 pooling layers; V2–V4 = simpler versions; 100 epochs; raw and black/white input",
         "VGG19, InceptionV3, ResNet50, MobileNet; 3 classes"],
    ], 0.6, 1.35, [1.6, 2.3, 2.5, 3.1, 2.6], row_h=0.45, center_from=99)
    text(s, "Measured: validation accuracy and training time.", 0.6, 5.95, 12, 0.45, color=MUTED)
    takeaway(s, "does a more complex model really help on simple drawings?")
    notes(s, "The original team compared four model families, from simple to complex. Logistic regression draws "
             "straight lines between classes and is the fast baseline. An SVM looks for the widest gap between "
             "classes, and its kernels allow curved boundaries. A CNN slides small filters over the image to find "
             "strokes and shapes; V1 is their full network and V2 to V4 are simpler versions. Transfer learning "
             "reuses networks trained on over a million photos.")

    # 3
    s = new()
    title(s, "We rebuilt their exact pipeline")
    steps = ["QuickDraw\n20,000 per class", "One fixed split\n16k / 2k / 2k,\nshared by every model",
             "Same preprocessing\nraw / black-white /\nscaled [−1, 1] / resized",
             "Same models\nand settings", "Measure\nvalidation accuracy\n+ training time"]
    bw, gap = 2.15, 0.33
    for i, st in enumerate(steps):
        bx = 0.6 + i * (bw + gap)
        box(s, st, bx, 1.35, bw, 1.75)
        if i:
            arrow(s, bx - gap + 0.03, 2.225, bx - 0.03, 2.225)
    tl_min = CROSS[CROSS.fam == "TL"].time.sum() / 60
    ref_s = sum(int(tl[m]["Reported training time (s)"]) * e for m, e in TL_EPOCHS.items())
    text(s, "What differed", 0.6, 3.4, 12, 0.45, bold=True, color=ACCENT)
    bullets(s, [
        "Newer hardware (RTX 4070 laptop GPU) and newer library versions",
        f"Transfer learning: frozen-network features computed once, identical result",
        f"{f1(tl_min)} min instead of about {f1(ref_s / 3600)} h with their method",
        "We read the authors' original code to explain any gaps",
    ], 0.6, 3.9, 12.1, 2.4, space=8)
    takeaway(s, "same experiment, so any difference must have a cause.")
    arith = " + ".join(f"{m} {tl[m]['Reported training time (s)']} s × {e}" for m, e in TL_EPOCHS.items())
    notes(s, "We rebuilt their pipeline step by step. The same QuickDraw data, one fixed split shared by every "
             "model, the same preprocessing for each model, the same models and settings, and the same two "
             "measurements. What differed: we used a newer laptop GPU and newer libraries. For transfer learning we "
             "computed the frozen networks' outputs once and trained only the new head, which gives the same "
             f"result in {f1(tl_min)} minutes instead of about {f1(ref_s / 3600)} hours.",
          f"Their time = report Table 6 seconds per epoch × epochs in their logs: {arith} = {ref_s:,} s "
          f"= {f1(ref_s / 3600)} h. Ours = sum of transfer-learning training times in cross_model_ours.csv "
          f"(feature extraction + head) = {int(CROSS[CROSS.fam == 'TL'].time.sum()):,} s = {f1(tl_min)} min.")

    # 4
    s = new()
    title(s, "Most results reproduce; three don't")
    st = lambda ok, why: (("Matches" if ok else "Differs") + (f": {why}" if why else ""), GREEN if ok else RED, True)
    lr_gap = max(abs(float(r["Difference (points)"])) for r in lr)
    others = ["ResNet50", "InceptionV3", "MobileNet"]
    table(s, [
        ["Model (classes)", "Reported", "Ours", "Status"],
        ["Logistic Regression (3 / 10 / 50)", " / ".join(f1(r["Reported accuracy (%)"]) for r in lr),
         " / ".join(f1(r["Our accuracy (%)"]) for r in lr), st(True, f"within {f1(lr_gap)} points")],
        ["CNN V1 (50)", f1(v1["Reported accuracy (%)"]), f1(v1["Our accuracy (%)"]), st(matches(v1), "")],
        ["VGG19 transfer learning (3)", f1(tl["VGG19"]["Reported accuracy (%)"]),
         f1(tl["VGG19"]["Our accuracy (%)"]), st(matches(tl["VGG19"]), "")],
        ["SVM RBF (3)", f1(rbf3["Reported accuracy (%)"]), f1(rbf3["Ours at 20k per class (%)"]),
         st(False, "ours higher")],
        ["CNN \"V2\" (50)", f1(v2["Reported accuracy (%)"]), f1(v2["Our accuracy (%)"]), st(False, "ours lower")],
        ["ResNet50 / InceptionV3 / MobileNet (3)", " / ".join(f1(tl[m]["Reported accuracy (%)"]) for m in others),
         " / ".join(f1(tl[m]["Our accuracy (%)"]) for m in others), st(False, "ours much higher")],
    ], 0.6, 1.3, [4.3, 2.1, 2.1, 3.6], row_h=0.6, center_from=1)
    text(s, "Validation accuracy (%), as in the paper.", 0.6, 5.65, 12, 0.45, color=MUTED)
    takeaway(s, "the reproduction works; three gaps needed explaining.")
    full = []
    for fam, d in REPRO.items():
        full.append(f"{fam}: " + "; ".join(
            f"{r['Model']} {r['Input']} {r['Classes']} cls {f1(r['Reported accuracy (%)'])} vs "
            f"{f1(r['Our accuracy (%)'])}" for _, r in d.iterrows()))
    notes(s, "Here is the scorecard, all in validation accuracy. Logistic regression matches at every class size, "
             f"within {f1(lr_gap)} points. Our CNN V1 matches their V1 on 50 classes, and VGG19 transfer learning "
             "matches almost exactly. So the pipeline is right. But three results differ. Our SVM with the RBF "
             "kernel scores much higher than theirs. Their CNN called V2 scores much higher than our V2. And three "
             "of the four transfer-learning networks score far higher for us. Because the setup is the same, each "
             "of these gaps needs a cause, which is the next slide. The full tables are in the repository.",
          "\n".join(full))

    # 5
    s = new()
    title(s, "Why the three differences happen")
    table(s, [
        ["What we saw", "Why", "Our evidence"],
        ["SVM scores higher than reported",
         "Their runs used ~70,000 drawings per class with the same 500-step limit: less converged",
         f"Our 70,000 rerun: RBF {f1(rbf3['Ours at 20k per class (%)'])} → {f1(rbf3['Ours at 70k per class (%)'])}"
         f" (3 classes), {f1(rbf10['Ours at 20k per class (%)'])} → {f1(rbf10['Ours at 70k per class (%)'])} (10); "
         f"theirs {f1(rbf3['Reported accuracy (%)'])} / {f1(rbf10['Reported accuracy (%)'])}"],
        ["Their \"V2\" equals our V1", "A script bug trained V1 under every V2–V4 label",
         f"Our V1 {f1(v1['Our accuracy (%)'])} vs their \"V2\" {f1(v2['Reported accuracy (%)'])}; "
         f"our real V2 {f1(v2['Our accuracy (%)'])}"],
        ["Transfer learning far higher for 3 of 4 networks",
         "Frozen BatchNorm layers behaved differently in 2018 Keras",
         f"VGG19 (no BatchNorm) matches: {f1(tl['VGG19']['Our accuracy (%)'])} vs "
         f"{f1(tl['VGG19']['Reported accuracy (%)'])}; their Inception log: {INCEPTION_TRAIN_ACC}% train vs "
         f"{f1(tl['InceptionV3']['Reported accuracy (%)'])}% validation"],
    ], 0.6, 1.2, [2.3, 3.0, 3.4], row_h=0.5, center_from=99, first_col_bold=True)
    picture(s, os.path.join(FIG, "tl_reported_vs_ours.png"), 9.15, 1.2, 4.1, 3.9)
    text(s, f"CNN V3 is {f1(-float(v3_10['Difference (points)']))} / {f1(-float(v3_50['Difference (points)']))} "
            "points lower at 10 / 50 classes: unexplained, likely seed or library differences.",
         0.6, 5.65, 12.1, 0.7, color=MUTED)
    takeaway(s, "the paper's claim that complex networks don't help doodles is likely an artefact.")
    notes(s, "Each gap has a cause we could check. First, the SVM. Their logs show about 70,000 drawings per class, "
             "with the same 500-step limit. More data under the same limit means a less finished model. When we "
             f"reran with 70,000 drawings, RBF dropped from {f1(rbf3['Ours at 20k per class (%)'])} to "
             f"{f1(rbf3['Ours at 70k per class (%)'])}, close to theirs. Second, their script had a bug that trained "
             "V1 under the V2 label, so their V2 is really our V1. Third, BatchNorm layers behaved differently in "
             "2018 Keras. VGG19, the only network without them, matches. So the claim that complex networks don't "
             "help is likely wrong.",
          f"Inception training accuracy {INCEPTION_TRAIN_ACC}% is from the authors' GitHub README log.")

    # 6
    s = new()
    title(s, "Our extension: XGBoost, a model family the paper never tried")
    y = 1.35
    labels = ["Tree 1", "Tree 2", "…", "Tree N"]
    for i, lab in enumerate(labels):
        if lab == "…":
            text(s, lab, 0.7, y, 2.1, 0.5, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        else:
            box(s, lab, 0.7, y, 2.1, 0.5, title_line=False)
        if i < len(labels) - 1:
            arrow(s, 1.75, y + 0.52, 1.75, y + 0.78)
        y += 0.8
    text(s, "Each new tree fixes the mistakes of the previous ones; their votes are added up.",
         0.6, y, 2.3, 2.0)
    bullets(s, [
        "Why: the reference had no tree-based model; it trains fast on a GPU",
        "Tuned 7 settings with Optuna Bayesian search, on validation only",
        "Test set used once; first run was time-limited, so we retrained fully",
    ], 3.0, 1.35, 9.7, 1.7, space=6)
    rows = [["Classes", "First run\n(time-limited)\nvalidation / test", "Fully trained\n\nvalidation / test"]]
    for k in (3, 10, 50):
        r = xgb(k)
        rows.append([str(k), f"{f1(r['Capped validation (%)'])} / {f1(r['Capped test (%)'])}",
                     f"{f1(r['Extended validation (%)'])} / {f1(r['Extended test (%)'])}"])
    table(s, rows, 3.0, 3.2, [1.25, 2.3, 2.3], row_h=0.5)
    picture(s, os.path.join(FIG, "xgb_50_learning_curve.png"), 8.9, 3.2, 3.85, 2.9)
    gain = f1(xgb(50)["Test gain (points)"])
    takeaway(s, f"more training gained only +{gain} points (test); XGBoost hits a ceiling around 78%.")
    notes(s, "For our extension we added XGBoost, because the paper never tried a tree-based model. XGBoost builds trees one after another; each new tree fixes the mistakes of the "
             "ones before, and their votes are added up. We tuned seven settings with Optuna, a Bayesian search, "
             "using only the validation set. Our first run was time-limited, so we "
             "retrained until the model stopped improving. On 50 classes that gained only half a point: the curve "
             "flattens around 78 percent.")

    # 7
    s = new()
    title(s, "All models compared: the CNN wins on 50 classes, accurately and fast")
    picture(s, os.path.join(FIG, "acc_vs_time_50_classes.png"), 0.6, 1.3, 12.1, 2.55)
    fams = [("LR", "Logistic Regression"), ("SVM", "SVM"), ("CNN", "CNN"), ("TL", "Transfer learning"),
            ("XGB", "XGBoost")]
    rows = [["Classes"] + [lab.replace(" ", "\n", 1) if lab.count(" ") else lab for _, lab in fams]]
    for k in (3, 10, 50):
        rows.append([str(k)] + ["—" if (b := best(f, k)) is None else f"{f1(b.acc)}% · {t_read(b.time)}"
                                for f, _ in fams])
    table(s, rows, 0.6, 3.9, [1.2, 1.9, 2.25, 2.2, 2.2, 2.35], row_h=0.42, size=BODY)
    text(s, "Best model per family, validation %. — = not run in the reference (SVM: 3 and 10 classes; "
            "transfer learning: 3 classes).", 0.6, 6.05, 12.1, 0.35, size=14, color=MUTED)
    takeaway(s, "simple models fall off as classes grow; CNNs scale best.")
    notes(s, "This chart shows all our models on 50 classes: accuracy upwards, training time to the right on a "
             f"log scale. The CNNs sit top left: accurate and quick. The best, CNN V1, reaches {f1(c50.acc)} percent "
             f"in about {t_read(c50.time)}. XGBoost reaches {f1(x50.acc)} percent but needs about "
             f"{t_read(x50.time)}. The table below gives the best "
             "model per family at each class size: every family does well on 3 classes, and the gaps open as the "
             "number of classes grows.")

    # 8
    s = new()
    title(s, "What we learned")
    c3, x3, c10, x10 = best("CNN", 3), best("XGB", 3), best("CNN", 10), best("XGB", 10)
    bullets(s, [
        f"CNN is best on 50 classes: {f1(c50.acc)}%, about {f1(speed)}× faster than XGBoost ({f1(x50.acc)}%)",
        "XGBoost matches CNNs on 3 and 10 classes; trees judge pixels, CNNs learn shapes",
        "Three surprising paper results traced to their setup: data size, script bug, BatchNorm",
        "Future work: feed transfer-learning features into XGBoost",
    ], 0.6, 1.4, 12.1, 4.6, space=20)
    takeaway(s, f"yes: a small CNN names the right doodle out of 50 classes {f1(c50.acc)}% of the time.")
    notes(s, f"To sum up. The CNN is the best model on 50 classes, at {f1(c50.acc)} percent, and it trains about "
             f"{f1(speed)} times faster than tuned XGBoost. XGBoost keeps up with CNNs on 3 and 10 classes, "
             f"{f1(x3.acc)} against {f1(c3.acc)} and {f1(x10.acc)} against {f1(c10.acc)}, but stalls on 50, because "
             "trees judge single pixels while CNNs learn shapes. The three surprising results in the paper come from "
             "their setup, not from the models. "
             "So yes, a computer can recognise doodles. Thank you.")

    for i, sl in enumerate(slides, 1):
        footer(sl, i)
    assert len(slides) <= 8
    prs.save(OUT)
    print("saved", OUT, f"({len(slides)} slides)")
    print("seconds per slide:", SECONDS, f"total {sum(SECONDS)} s = {f1(sum(SECONDS) / 60)} min")
    print("missing figures:", MISSING or "none")


if __name__ == "__main__":
    build()

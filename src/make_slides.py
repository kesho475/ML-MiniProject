"""Phase 10: build the review deck (presentation/ML_MiniProject.pptx) from the saved results.

Run: python src/make_slides.py
Every number comes from results/tables/*.csv, read at build time and shown with one decimal (half-up).
Images come from results/figures/phase9/. A missing figure becomes a marked [PLACEHOLDER ...] box and
is listed when the script finishes; numbers that are not in results/ are marked [PLACEHOLDER ...] in the text.
"""
import os
from decimal import ROUND_HALF_UP, Decimal

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
FIG = os.path.join(ROOT, "results", "figures", "phase9")
OUT = os.path.join(ROOT, "presentation", "ML_MiniProject.pptx")

FONT = "Arial"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)  # dark navy: titles and table headers
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ROW_ALT = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FOOTER = "Drawing: A New Way To Search — Reproduce & Extend"
BODY = 18
MISSING = []


# ---------- numbers ----------
def f1(x):
    """One decimal, half-up, from the value as written in the CSV."""
    return f"{Decimal(repr(float(x))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):,}"


rep = pd.concat(pd.read_csv(os.path.join(TAB, f"reproduction_{f}.csv")) for f in ["LR", "SVM", "CNN", "TL"])
svm = pd.read_csv(os.path.join(TAB, "svm_reported_20k_70k.csv"))
cm = pd.read_csv(os.path.join(TAB, "cross_model_ours.csv"))
xgb = pd.read_csv(os.path.join(TAB, "xgb_capped_vs_extended.csv"), dtype={"classes": str})


def R(family, model, inp, k, split="dev"):
    r = rep[(rep.family == family) & (rep.model == model) & (rep.input == inp) & (rep.classes == k) & (rep.split == split)]
    assert len(r) == 1, (family, model, inp, k)
    return r.iloc[0]


def S(model, k):
    return svm[(svm.model == model) & (svm.classes == k)].iloc[0]


def X(k, label):
    return xgb[(xgb.classes == str(k)) & (xgb.label == label)].iloc[0]


def best(family, k):
    d = cm[(cm.family == family) & (cm.classes == k)]
    return None if d.empty else d.loc[d.dev_acc.idxmax()]


# ---------- drawing helpers ----------
def _font(run, size=BODY, bold=False, color=INK):
    run.font.name, run.font.size, run.font.bold, run.font.color.rgb = FONT, Pt(size), bold, color


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


def bullets(slide, items, x, y, w, h, size=BODY, space=10):
    assert len(items) <= 5, "at most 5 bullets per slide"
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.3)))
        pPr.set("indent", str(-Inches(0.3)))
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
        r = p.add_run()
        r.text = item
        _font(r, size)


def title(slide, s, size=30):
    text(slide, s, 0.6, 0.3, 12.1, 0.8, size=size, bold=True, color=ACCENT, anchor=MSO_ANCHOR.MIDDLE)


def footer(slide, n):
    text(slide, FOOTER, 0.6, 7.05, 10, 0.3, size=10, color=MUTED)
    text(slide, str(n), 11.7, 7.05, 1.0, 0.3, size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def picture(slide, name, x, y, w, h):
    """Fit the figure inside the box (keeps aspect ratio, centred). Missing -> marked placeholder box."""
    path = os.path.join(FIG, name)
    if not os.path.isfile(path) or os.path.islink(path):
        MISSING.append(f"figure {name}")
        b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        b.fill.background()
        b.line.color.rgb = MUTED
        text(slide, f"[PLACEHOLDER: missing figure {name}]", x, y + h / 2 - 0.3, w, 0.6, align=PP_ALIGN.CENTER)
        return
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, rows, x, y, col_w, row_h=0.4, size=BODY, center_from=1):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(col_w)), Inches(row_h * len(rows)))
    t = shape.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = ACCENT if i == 0 else (ROW_ALT if i % 2 == 0 else WHITE)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            for k, line in enumerate(str(val).split("\n")):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                p.alignment = PP_ALIGN.CENTER if j >= center_from else PP_ALIGN.LEFT
                r = p.add_run()
                r.text = line
                _font(r, size, bold=(i == 0), color=WHITE if i == 0 else INK)


def box(slide, s, x, y, w, h, size=BODY):
    b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = WHITE
    b.line.color.rgb = ACCENT
    b.line.width = Pt(1.5)
    b.shadow.inherit = False
    tf = b.text_frame
    tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = s
    _font(r, size)


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = ACCENT
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle"}))


def notes(slide, s):
    words = len(s.split())
    assert 65 <= words <= 95, f"speaker notes have {words} words"
    slide.notes_slide.notes_text_frame.text = s


# ---------- slides ----------
def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    slides = []

    def new():
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        sl.background.fill.solid()
        sl.background.fill.fore_color.rgb = WHITE
        slides.append(sl)
        return sl

    # 1. Problem and dataset
    s = new()
    title(s, "Drawing: A New Way To Search — Reproduce & Extend")
    text(s, "[Name 1, SRN] · [Name 2, SRN]", 0.6, 1.1, 8, 0.45, color=MUTED)
    bullets(s, [
        "Goal: classify hand-drawn doodles; the reference's motivation is drawing as a search interface",
        "Data: Google QuickDraw, 28×28 grayscale bitmaps",
        "3 / 10 / 50 classes, 20,000 drawings per class",
        "Split per class 80/10/10: 16,000 / 2,000 / 2,000 (train / validation / test)",
    ], 0.6, 1.8, 7.4, 5.0)
    picture(s, "samples_10x5.png", 8.3, 1.2, 4.5, 5.7)
    notes(s, "Our project reproduces and extends the report Drawing: A New Way To Search. The task is to classify "
             "hand-drawn doodles, and the authors' motivation is that a sketch could work as a search query. The data "
             "is Google QuickDraw: 28 by 28 grayscale bitmaps. Like the report, we use 3, 10 and 50 classes with "
             "20,000 drawings per class, split 80, 10, 10 per class into training, validation and test. On the "
             "right you can see five drawings from each of the ten classes in the 10-class set.")

    # 2. Reference approach
    s = new()
    title(s, "Reference approach (the original paper)")
    table(s, [
        ["Family", "Models", "Why it was used", "Key settings"],
        ["Logistic Regression", "LR", "Simple, fast baseline", "lbfgs, 100 iterations, raw pixels"],
        ["SVM", "Linear, RBF, poly (deg 5), sigmoid", "Tests kernel non-linearity",
         "500 iterations, inputs scaled to [−1, 1], 3 and 10 classes only"],
        ["CNN", "V1–V4", "Learns spatial patterns",
         "V1 = 2 conv + 2 max-pool + Dense 128, simplified to V2–V4; Adam, batch 500, 100 epochs; "
         "raw and binarized inputs"],
        ["Transfer learning", "VGG19, InceptionV3, ResNet50, MobileNet", "Reuses ImageNet knowledge",
         "Frozen base + small head, 3 classes"],
    ], 0.6, 1.25, [2.2, 2.9, 2.6, 4.4], row_h=0.5, center_from=99)
    text(s, "Metrics: validation accuracy and training time (report §5.1).", 0.6, 6.45, 12, 0.45, color=MUTED)
    notes(s, "The reference compares four model families. Logistic regression is the simple, fast baseline. SVMs "
             "test whether non-linear kernels help, with four kernels capped at 500 iterations, and only on 3 and 10 "
             "classes. The CNNs learn spatial patterns: V1 has two convolution and pooling blocks, and V2 to V4 "
             "simplify it step by step, on raw and binarized input. Transfer learning reuses ImageNet networks with a "
             "frozen base and a small head, on 3 classes. The report measures validation accuracy and training time.")

    # 3. Our reproduction setup
    s = new()
    title(s, "Our reproduction setup")
    tl_min = cm[cm.family == "TL"].time_s.sum() / 60  # feature extraction + head training, all four networks
    bullets(s, [
        "Same data, split, models, hyperparameters and metrics as the report",
        "One fixed train / validation / test split, shared by every model",
        "RTX 4070 Laptop 8 GB, i7-14650HX; Python 3.12.14, TensorFlow 2.21.0 (Keras 3.15.1), "
        "scikit-learn 1.9.1, XGBoost 3.4.1",
        f"Transfer learning on features precomputed once by the frozen networks: same maths as frozen training, "
        f"{f1(tl_min)} min in total instead of [PLACEHOLDER: ~16 GPU-hours, not in results/]",
        "We also read the authors' original code to explain differences",
    ], 0.6, 1.3, 12.1, 4.2)
    for i, lab in enumerate(["Data", "Split", "Model", "Metrics"]):
        bx = 1.6 + i * 2.75
        box(s, lab, bx, 5.75, 1.9, 0.75)
        if i:
            arrow(s, bx - 0.85, 6.125, bx - 0.02, 6.125)
    notes(s, "We kept everything the report specifies: the same data, models, hyperparameters and metrics. One fixed "
             "split is saved once and shared by every model, so all comparisons are fair. We ran on a laptop RTX 4070 "
             "with current library versions, which is one source of differences. For transfer learning we computed the "
             "frozen networks' features once and trained only the head; that is mathematically the same, and took "
             f"about {f1(tl_min)} minutes. We also read the authors' code to explain the gaps.")

    # 4. Reproduced results: LR, SVM, CNN
    s = new()
    title(s, "Reproduced results: LR, SVM, CNN (validation %)")
    rows = [["Model", "Classes", "Reported", "Ours", "Ours, 70k"]]
    for k in (3, 10, 50):
        r = R("LR", "logreg", "raw", k)
        rows.append(["LR", k, f1(r.rep_acc), f1(r.our_acc), "—"])
    for m, lab in [("rbf", "SVM RBF"), ("poly", "SVM poly")]:
        for k in (3, 10):
            r = S(m, k)
            rows.append([lab, k, f1(r.rep_acc), f1(r.acc_20k), f1(r.acc_70k)])
    v1, v2, v3 = R("CNN", "V1", "raw", 50), R("CNN", "V2", "raw", 50), R("CNN", "V3", "binarized", 50)
    v3_10 = R("CNN", "V3", "binarized", 10)
    rows += [["CNN V1 raw", 50, f1(v1.rep_acc), f1(v1.our_acc), "—"],
             ["CNN V2 raw", 50, f1(v2.rep_acc), f1(v2.our_acc), "—"],
             ["CNN V3 binarized", 50, f1(v3.rep_acc), f1(v3.our_acc), "—"]]
    table(s, rows, 0.6, 1.25, [2.5, 1.3, 1.3, 1.1, 1.4], row_h=0.44)
    lr_gap = rep[rep.family == "LR"]["diff"].abs().max()
    bullets(s, [
        f"LR within {f1(lr_gap)} points at all class sizes",
        "SVM RBF / poly above the report: its runs used ~70,000 drawings per class under the same 500-iteration "
        "cap; our 70k rerun moves them back toward it",
        f"Our CNN V1 ({f1(v1.our_acc)}) matches the reported \"V2\" ({f1(v2.rep_acc)}); our true V2 gets "
        f"{f1(v2.our_acc)}: the original code ran V1 under the V2 label",
        f"CNN V3: {f1(-v3_10['diff'])} / {f1(-v3['diff'])} points below at 10 / 50 classes, unexplained",
    ], 8.5, 1.25, 4.3, 5.6)
    notes(s, f"Logistic regression reproduces within about {f1(lr_gap)} points. The SVMs with RBF and polynomial "
             f"kernels score far above the report, for example {f1(S('rbf', 3).acc_20k)} against "
             f"{f1(S('rbf', 3).rep_acc)}. The authors' logs imply about 70,000 drawings per class, and with the same "
             f"500-iteration cap more data means less convergence; our 70k rerun drops RBF to "
             f"{f1(S('rbf', 3).acc_70k)}. For CNNs, our V1 matches their V2, while our real V2 is much lower: their "
             f"code overwrote V2 with V1. The V3 gap remains unexplained.")

    # 5. Reproduced results: transfer learning
    s = new()
    title(s, "Reproduced results: transfer learning (3 classes)")
    picture(s, "tl_batchnorm.png", 0.6, 1.25, 7.4, 5.6)
    vgg, mob, res, inc = (R("TL", m, "resized", 3) for m in ["vgg19", "mobilenet", "resnet50", "inception"])
    bullets(s, [
        f"VGG19 (no BatchNorm) matches: {f1(vgg.our_acc)} vs {f1(vgg.rep_acc)} reported",
        f"BatchNorm networks far higher: MobileNet {f1(mob.our_acc)} vs {f1(mob.rep_acc)}, ResNet50 "
        f"{f1(res.our_acc)} vs {f1(res.rep_acc)}, InceptionV3 {f1(inc.our_acc)} vs {f1(inc.rep_acc)}",
        f"Cause: frozen BatchNorm behaved differently in 2018 Keras; their Inception log shows "
        f"[PLACEHOLDER: ~100% training, not in results/] vs {f1(inc.rep_acc)}% validation",
        "So \"complex networks don't help doodles\" is likely an artefact",
    ], 8.3, 1.25, 4.5, 5.6)
    notes(s, f"VGG19 has no BatchNorm layers, and it reproduces almost exactly: {f1(vgg.our_acc)} against "
             f"{f1(vgg.rep_acc)}. The three networks with BatchNorm score far higher for us, up to "
             f"{f1(inc.our_acc)} against {f1(inc.rep_acc)} for Inception. In 2018 Keras, frozen BatchNorm layers "
             "still used batch statistics during training, so training and validation behaved differently; their "
             "own Inception log shows near-perfect training accuracy with poor validation accuracy. So the report's "
             "conclusion that complex networks do not help doodles is most likely an artefact of that behaviour.")

    # 6. Extension: XGBoost
    s = new()
    title(s, "Extension: XGBoost (gradient-boosted trees)")
    y = 1.35
    labels = ["Tree 1", "Tree 2", "…", "Tree N", "Sum → prediction"]
    for i, lab in enumerate(labels):
        if lab == "…":
            text(s, lab, 0.9, y, 2.6, 0.55, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        else:
            box(s, lab, 0.9, y, 2.6, 0.55)
        if i < len(labels) - 1:
            arrow(s, 2.2, y + 0.57, 2.2, y + 0.83)
        y += 0.85
    text(s, "Each tree corrects the errors of the trees before it", 0.6, y + 0.05, 3.2, 1.0)
    picture(s, "xgb50_capped_vs_converged_curve.png", 4.0, 1.25, 8.8, 3.35)
    t = {k: (X(k, "xgb_capped").test_acc, X(k, "xgb_extended").test_acc) for k in (3, 10, 50)}
    bullets(s, [
        "Why: the reference had no tree-based model, so a fourth model family",
        # trial counts: DECISIONS.md Phase 8 (xgb_3 45, xgb_10 31, xgb_50 top 4) and Phase 8b (xgb10_wide 32)
        "Optuna (TPE, Bayesian) on validation: 45 / 31 + 32 trials at 3 / 10 classes; top 4 settings on 50",
        "Phase 8 time-capped; 8B uncapped, wider search. Test % at 3 / 10 / 50: " +
        " / ".join(f1(a) for a, _ in t.values()) + " → " + " / ".join(f1(b) for _, b in t.values()),
        "Every 50-class setting plateaued at about 78%",
    ], 4.0, 4.7, 8.8, 2.3, space=4)
    notes(s, "As our extension we added XGBoost, because the reference has no tree-based model. Boosting adds trees "
             "one at a time, each correcting the errors of the previous ones. We tuned it with Optuna, a Bayesian "
             "optimiser, on the validation set. Our first run was time-capped at 50 classes; the extended run removed "
             f"the cap and widened the search. Test accuracy moved from {f1(t[50][0])} to {f1(t[50][1])} at 50 "
             "classes. The curves show every 50-class setting levelling off at about 78 percent.")

    # 7. Comparison of all models
    s = new()
    title(s, "Comparison of all models (our runs, validation accuracy)")
    picture(s, "accuracy_vs_time.png", 0.6, 1.1, 12.1, 3.85)
    fams = [("LR", "LR"), ("SVM", "SVM"), ("CNN", "CNN"), ("TL", "TL"), ("XGB", "XGBoost")]
    name = {"logreg": "LR", "rbf": "RBF", "poly": "poly", "xgb": "Phase 8", "xgb_extended": "Phase 8B",
            "resnet50": "ResNet50", "vgg19": "VGG19", "mobilenet": "MobileNet", "inception": "InceptionV3"}
    head = ["Classes"]
    for fam, lab in fams:  # name the best model in the header when it is the same at every class size
        ms = {f"{b.model} {b.input}" if fam == "CNN" else name.get(b.model, b.model)
              for b in (best(fam, k) for k in (3, 10, 50)) if b is not None}
        head.append(f"{lab} ({ms.pop()})" if len(ms) == 1 and fam not in ("LR", "XGB") else
                    lab if fam in ("LR", "XGB") else f"{lab} (best)")
    rows = [head] + [[str(k)] + ["—" if (b := best(fam, k)) is None else f"{f1(b.dev_acc)}% · {f1(b.time_s)} s"
                                 for fam, _ in fams] for k in (3, 10, 50)]
    table(s, rows, 0.6, 5.05, [1.2, 1.9, 2.0, 2.0, 2.3, 2.7], row_h=0.42)
    c50, x50 = best("CNN", 50), best("XGB", 50)
    notes(s, "This chart puts every model on one accuracy against training-time plot, with a log time axis, one panel "
             f"per class size. The table gives the best model of each family. At 50 classes the CNN leads with "
             f"{f1(c50.dev_acc)} percent in {f1(c50.time_s)} seconds, while tuned XGBoost reaches {f1(x50.dev_acc)} "
             f"in {f1(x50.time_s)} seconds. Transfer learning is the most accurate at 3 classes. Logistic regression "
             "is the fastest but falls off quickly as classes increase, and the SVMs sit between it and the CNNs.")

    # 8. Key findings
    s = new()
    title(s, "Key findings")
    c3, x3, c10, x10 = best("CNN", 3), best("XGB", 3), best("CNN", 10), best("XGB", 10)
    speed = x50.time_s / c50.time_s
    bullets(s, [
        f"CNN is best on 50 classes: {f1(c50.dev_acc)}% in {f1(c50.time_s)} s, about {f1(speed)}× faster than "
        f"tuned XGBoost ({f1(x50.dev_acc)}%, {f1(x50.time_s)} s)",
        f"XGBoost matches CNNs on 3 and 10 classes ({f1(x3.dev_acc)} vs {f1(c3.dev_acc)}; {f1(x10.dev_acc)} vs "
        f"{f1(c10.dev_acc)}) but plateaus at about 78% on 50: trees judge single pixels, CNNs learn shapes",
        "Three inconsistencies in the reference explained: the CNN version bug, the SVM data size, "
        "the frozen BatchNorm effect",
        "Future work: feed transfer-learning features into XGBoost or LR, as the report suggests",
    ], 0.6, 1.4, 12.1, 5.4, space=16)
    notes(s, f"Four findings. First, the CNN is the best model at 50 classes and about {f1(speed)} times faster to "
             "train than tuned XGBoost. Second, XGBoost keeps up with CNNs on 3 and 10 classes but stalls near 78 "
             "percent on 50, because trees look at individual pixels while convolutions learn shapes. Third, three "
             "surprising results in the reference have clear explanations: a version bug, a larger SVM dataset, and "
             "frozen BatchNorm. Finally, a natural next step is feeding transfer-learning features into XGBoost.")

    for i, sl in enumerate(slides, 1):
        footer(sl, i)
    assert len(slides) <= 8
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print("saved", OUT, f"({len(slides)} slides)")
    print("missing figures:", MISSING or "none")


if __name__ == "__main__":
    build()

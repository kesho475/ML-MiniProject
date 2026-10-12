"""Fixed train/dev/test split (build once) and model-specific loaders.

Only this module reads/writes data/processed/*.npz. Run `python src/data.py` to
build qd_3/10/50.npz, verify them and save the sample-drawing grids.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from classes import CLASS_SETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
N_PER_CLASS = 20_000
SPLITS = ("train", "dev", "test")


def _path(k, n=N_PER_CLASS):
    """qd_<k>.npz for the main 20k-per-class data; qd<n/1000>k_<k>.npz for the optional svm70k data."""
    return os.path.join(PROC, f"qd_{k}.npz" if n == N_PER_CLASS else f"qd{n // 1000}k_{k}.npz")


def _split_class(name, n=N_PER_CLASS):
    """First n rows of the class -> 80% train / 10% dev / 10% test (seed 42, as massageData.py)."""
    x = np.load(os.path.join(RAW, name + ".npy"), mmap_mode="r")[:n]
    tr, ev = train_test_split(x, test_size=0.2, random_state=42)
    dev, te = train_test_split(ev, test_size=0.5, random_state=42)
    return tr, dev, te


def build(k, n=N_PER_CLASS):
    """Save data/processed/qd_<k>.npz (see _path); labels follow the class order in classes.py."""
    names = CLASS_SETS[k]
    X = {s: [] for s in SPLITS}
    y = {s: [] for s in SPLITS}
    for i, name in enumerate(names):
        for s, x in zip(SPLITS, _split_class(name, n)):
            X[s].append(x)
            y[s].append(np.full(len(x), i, dtype=np.int64))
    out = {"class_names": np.array(names)}
    for s in SPLITS:
        out[f"X_{s}"] = np.concatenate(X[s])
        out[f"y_{s}"] = np.concatenate(y[s])
    os.makedirs(PROC, exist_ok=True)
    np.savez(_path(k, n), **out)


def load(k, variant="raw", n=N_PER_CLASS):
    """Return (X_train, y_train, X_dev, y_dev, X_test, y_test, class_names).

    variant: 'uint8' (as saved) | 'raw' float32 0-255 | 'binarized' (>0 -> 255)
             | 'scaled' (MinMax to [-1,1], fit on train)
             | 'image' / 'binarized_image': same as raw / binarized, shaped N x 28 x 28 x 1
    """
    d = np.load(_path(k, n))
    X = {s: d[f"X_{s}"] for s in SPLITS}
    if variant in ("binarized", "binarized_image"):
        X = {s: np.where(x > 0, 255, 0).astype(np.float32) for s, x in X.items()}
    elif variant in ("raw", "image"):
        X = {s: x.astype(np.float32) for s, x in X.items()}
    elif variant == "scaled":
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(-1, 1)).fit(X["train"])
        X = {s: sc.transform(x) for s, x in X.items()}
    elif variant != "uint8":
        raise ValueError(variant)
    if variant.endswith("image"):
        X = {s: x.reshape(-1, 28, 28, 1) for s, x in X.items()}
    return (X["train"], d["y_train"], X["dev"], d["y_dev"], X["test"], d["y_test"],
            [str(n) for n in d["class_names"]])


def resized_batches(x_flat, size, batch=50):
    """Yield float32 (b, size, size, 3) batches from flat 784 rows; values stay 0-255.

    Batch by batch only: all images at once would need tens of GB (PLAN Phase 3).
    """
    import tensorflow as tf
    for i in range(0, len(x_flat), batch):
        b = x_flat[i:i + batch].reshape(-1, 28, 28, 1).astype(np.float32)
        b = tf.image.resize(b, (size, size)).numpy()
        yield np.repeat(b, 3, axis=-1)


def _verify(k, n=N_PER_CLASS):
    xtr, ytr, xdv, ydv, xte, yte, names = load(k, "uint8", n)
    for y, c in ((ytr, n * 8 // 10), (ydv, n // 10), (yte, n // 10)):
        assert (np.bincount(y, minlength=len(names)) == c).all(), "class counts"
    # No shared row between splits of a class. QuickDraw has a few exact-duplicate
    # drawings, so allow a tiny content overlap instead of exactly zero.
    for c, name in enumerate(names):
        s = [{r.tobytes() for r in x[y == c]} for x, y in ((xtr, ytr), (xdv, ydv), (xte, yte))]
        ov = len(s[0] & s[1]) + len(s[0] & s[2]) + len(s[1] & s[2])
        assert ov < 50 * n // N_PER_CLASS, (name, ov)
    print(f"qd_{k}: OK {xtr.shape} {xdv.shape} {xte.shape}")


def _check_variants():
    xr = load(3, "raw")[0]
    xb = load(3, "binarized")[0]
    xs = load(3, "scaled")[0]
    xi = load(3, "image")[0]
    assert xr.dtype == np.float32 and xr.min() == 0 and xr.max() == 255
    assert set(np.unique(xb)) == {0.0, 255.0}
    assert np.isclose(xs.min(), -1) and np.isclose(xs.max(), 1)
    assert xi.shape[1:] == (28, 28, 1)
    print("variants OK")


def _plot_samples():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xtr, ytr, *_, names = load(50, "uint8")
    out = os.path.join(ROOT, "results", "figures", "samples")
    os.makedirs(out, exist_ok=True)
    for c, name in enumerate(names):
        g = xtr[ytr == c][:25].reshape(5, 5, 28, 28).transpose(0, 2, 1, 3).reshape(140, 140)
        plt.imsave(os.path.join(out, f"{name}.png"), g, cmap="gray_r")


if __name__ == "__main__":
    import sys
    if "--70k" in sys.argv:  # optional svm70k data (Q4 check), 3 and 10 classes only
        for k in (3, 10):
            build(k, 70_000)
            _verify(k, 70_000)
    else:
        for k in (3, 10, 50):
            build(k)
            _verify(k)
        _check_variants()
        _plot_samples()

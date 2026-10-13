"""QuickDraw data: download the class files, build the fixed train/dev/test split, load it for each model.

    python src/data.py download        # 50 numpy_bitmap files into data/raw/
    python src/data.py split           # data/processed/qd_3.npz, qd_10.npz, qd_50.npz (20,000 per class)
    python src/data.py split --70k     # qd70k_3.npz, qd70k_10.npz (70,000 per class, for the SVM check)
"""
import argparse
import csv
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
N_PER_CLASS = 20_000
SPLITS = ("train", "dev", "test")

# Class sets from the reference's download scripts; labels follow this order.
CLASSES_3 = ["eraser", "belt", "banana"]
CLASSES_10 = ["laptop", "rain", "hockey stick", "wristwatch", "nose", "pond",
              "The Mona Lisa", "banana", "panda", "paint can"]
CLASSES_50 = ["squirrel", "paint can", "stitches", "lighter", "panda", "hockey stick",
              "kangaroo", "rain", "sea turtle", "aircraft carrier", "beard", "chair",
              "hammer", "hedgehog", "spreadsheet", "camel", "nose", "peas", "wristwatch",
              "banana", "belt", "eraser", "harp", "shark", "skull", "face", "castle",
              "drill", "lobster", "snowflake", "speedboat", "bee", "fence", "floor lamp",
              "hat", "leaf", "ambulance", "pond", "sweater", "grapes", "laptop",
              "screwdriver", "broccoli", "finger", "teddy-bear", "umbrella", "trumpet",
              "snake", "star", "The Mona Lisa"]
CLASS_SETS = {3: CLASSES_3, 10: CLASSES_10, 50: CLASSES_50}


# ---------- download ----------
def _rows(path):
    """Row count if the file is a valid (N, 784) uint8 array with N >= 20,000, else None."""
    try:
        a = np.load(path, mmap_mode="r")
        ok = a.ndim == 2 and a.shape[1] == 784 and a.dtype == np.uint8 and a.shape[0] >= N_PER_CLASS
        return a.shape[0] if ok else None
    except Exception:
        return None


def _fetch(name):
    path = os.path.join(RAW, name + ".npy")
    if _rows(path) is None:  # files that already verify are skipped
        urllib.request.urlretrieve(URL + urllib.parse.quote(name) + ".npy", path + ".part")
        os.replace(path + ".part", path)
    n = _rows(path)
    print(f"{name}: rows={n}", flush=True)
    return name, n, os.path.getsize(path) if n else 0


def download():
    os.makedirs(RAW, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(_fetch, CLASSES_50))
    with open(os.path.join(ROOT, "data", "download_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "rows", "bytes"])
        w.writerows(rows)
    print("done, bad files:", [r[0] for r in rows if r[1] is None])


# ---------- fixed split ----------
def _path(k, n=N_PER_CLASS):
    return os.path.join(PROC, f"qd_{k}.npz" if n == N_PER_CLASS else f"qd{n // 1000}k_{k}.npz")


def _split_class(name, n=N_PER_CLASS):
    """First n drawings of the class -> 80% train / 10% dev / 10% test (seed 42, as in the reference)."""
    x = np.load(os.path.join(RAW, name + ".npy"), mmap_mode="r")[:n]
    tr, ev = train_test_split(x, test_size=0.2, random_state=42)
    dev, te = train_test_split(ev, test_size=0.5, random_state=42)
    return tr, dev, te


def build(k, n=N_PER_CLASS):
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


def verify(k, n=N_PER_CLASS):
    xtr, ytr, xdv, ydv, xte, yte, names = load(k, "uint8", n)
    for y, c in ((ytr, n * 8 // 10), (ydv, n // 10), (yte, n // 10)):
        assert (np.bincount(y, minlength=len(names)) == c).all(), "class counts"
    # QuickDraw has a few exact duplicate drawings, so allow a tiny overlap between splits.
    for c, name in enumerate(names):
        s = [{r.tobytes() for r in x[y == c]} for x, y in ((xtr, ytr), (xdv, ydv), (xte, yte))]
        ov = len(s[0] & s[1]) + len(s[0] & s[2]) + len(s[1] & s[2])
        assert ov < 50 * n // N_PER_CLASS, (name, ov)
    print(f"{os.path.basename(_path(k, n))}: OK {xtr.shape} {xdv.shape} {xte.shape}")


# ---------- loading ----------
def load(k, variant="raw", n=N_PER_CLASS):
    """Return (X_train, y_train, X_dev, y_dev, X_test, y_test, class_names).

    variant: 'uint8' (as saved) | 'raw' float32 0-255 | 'binarized' (>0 -> 255)
             | 'scaled' (MinMax to [-1, 1], fit on train)
             | 'image' / 'binarized_image': raw / binarized shaped N x 28 x 28 x 1
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
    """Yield float32 (b, size, size, 3) batches from flat 784-pixel rows; values stay 0-255.
    Batch by batch, because all images at once would need tens of GB."""
    import tensorflow as tf
    for i in range(0, len(x_flat), batch):
        b = x_flat[i:i + batch].reshape(-1, 28, 28, 1).astype(np.float32)
        b = tf.image.resize(b, (size, size)).numpy()
        yield np.repeat(b, 3, axis=-1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "split"])
    ap.add_argument("--70k", dest="n70k", action="store_true", help="70,000 drawings per class, 3 and 10 classes")
    a = ap.parse_args()
    if a.step == "download":
        download()
    else:
        n = 70_000 if a.n70k else N_PER_CLASS
        for k in ((3, 10) if a.n70k else (3, 10, 50)):
            build(k, n)
            verify(k, n)

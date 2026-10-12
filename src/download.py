# Phase 2: download the 50 QuickDraw numpy_bitmap files, verify, write a log.
# Re-runnable: files that already verify are skipped.
import csv, os, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from classes import CLASSES_50

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def check(path):
    """Return row count if file is a valid (N,784) uint8 array with N>=20000, else None."""
    try:
        a = np.load(path, mmap_mode="r")
        return a.shape[0] if a.ndim == 2 and a.shape[1] == 784 and a.dtype == np.uint8 and a.shape[0] >= 20000 else None
    except Exception:
        return None

def fetch(name):
    path = os.path.join(RAW, name + ".npy")
    if check(path) is None:
        url = BASE + urllib.parse.quote(name) + ".npy"
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    n = check(path)
    print(f"{name}: rows={n}", flush=True)
    return name, n, os.path.getsize(path) if n else 0

os.makedirs(RAW, exist_ok=True)
with ThreadPoolExecutor(4) as ex:
    rows = list(ex.map(fetch, CLASSES_50))
with open(os.path.join(RAW, "..", "download_log.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["class", "rows", "bytes"]); w.writerows(rows)
bad = [r[0] for r in rows if r[1] is None]
print("DONE, bad files:", bad)

"""Phase 7 (Option A, step 1): run every image once through the frozen ImageNet base.
Usage: python src/tl_features.py <vgg19|mobilenet|resnet50|inception> [--smoke]

Saves data/features/<base>.npz (globally pooled features f_train/f_dev/f_test) and <base>_meta.json
(extract_time_s). Inception also saves the output of layer `mixed8` (= input of layer 249, the only tensor
crossing the stage-2 cut) as float16 memmaps for the fine-tuning stage.
Images: 28x28 -> 224 (299 Inception), 3 channels, values stay 0-255, no preprocess_input (spec).
--smoke: tiny strided subset, written to data/features/smoke/.
"""
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import numpy as np
import tensorflow as tf

from data import load, resized_batches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPS = {"vgg19": (keras.applications.VGG19, 224), "mobilenet": (keras.applications.MobileNet, 224),
        "resnet50": (keras.applications.ResNet50, 224), "inception": (keras.applications.InceptionV3, 299)}
BATCH = 50  # 100 ran out of GPU memory (8 GB) on VGG19 / Inception
SMOKE_N = {"train": 300, "dev": 100, "test": 100}


def feat_dir(smoke):
    d = os.path.join(ROOT, "data", "features", "smoke" if smoke else "")
    os.makedirs(d, exist_ok=True)
    return d


def main(base, smoke=False):
    out = feat_dir(smoke)
    meta_path = os.path.join(out, base + "_meta.json")
    if os.path.exists(meta_path):
        print(f"{base}: features already extracted ({meta_path})")
        return
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    app, size = APPS[base]
    net = app(weights="imagenet", include_top=False, input_shape=(size, size, 3))
    outputs = [keras.layers.GlobalAveragePooling2D()(net.output)]
    if base == "inception":
        outputs.append(net.layers[248].output)  # mixed8, 8x8x1280
    extractor = keras.Model(net.input, outputs)
    xtr, _, xdv, _, xte, _, _ = load(3, "uint8")
    t0 = time.time()
    pooled = {}
    for split, x in (("train", xtr), ("dev", xdv), ("test", xte)):
        if smoke:  # strided, so all 3 classes appear (rows are ordered by class)
            x = x[np.linspace(0, len(x) - 1, SMOKE_N[split]).astype(int)]
        mix = None
        if base == "inception":
            mix = np.lib.format.open_memmap(os.path.join(out, f"mixed8_{split}.npy"), mode="w+",
                                            dtype=np.float16, shape=(len(x), 8, 8, 1280))
        parts, i = [], 0
        for b in resized_batches(x, size, BATCH):
            o = extractor.predict_on_batch(b)  # inference mode: frozen base (Q7); compiled, so memory is freed early
            parts.append(o[0] if mix is not None else o)
            if mix is not None:
                mix[i:i + len(b)] = o[1].astype(np.float16)
            i += len(b)
        if mix is not None:
            mix.flush()
        pooled[split] = np.concatenate(parts)
        print(f"{base} {split}: {pooled[split].shape} min={pooled[split].min():.2f} max={pooled[split].max():.2f}")
    np.savez(os.path.join(out, base + ".npz"), f_train=pooled["train"], f_dev=pooled["dev"], f_test=pooled["test"])
    with open(meta_path, "w") as f:  # written last: its existence means extraction is complete
        json.dump({"extract_time_s": round(time.time() - t0, 1), "dim": int(pooled["train"].shape[1]),
                   "smoke": smoke}, f)
    print(f"{base}: extraction done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], "--smoke" in sys.argv)

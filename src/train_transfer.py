"""Transfer learning on 3 classes: frozen ImageNet network + small new head.

Step 1 runs every image once through the frozen network and saves its pooled features (skipped if they
exist). Step 2 trains the head, Dense(128, relu) -> Dense(3, softmax), on those features: the same result
as training with the frozen network attached, much faster. InceptionV3 then fine-tunes its top two blocks
(layers 249+) together with the head, on the saved output of layer `mixed8`.
Images: 28x28 -> 224 (299 for Inception), 3 channels, values stay 0-255.

    python src/train_transfer.py --base vgg19        # vgg19 | mobilenet | resnet50 | inception
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from common import ROOT, Run, add_out_arg, out, set_out
from data import load, resized_batches

APPS = {"vgg19": (keras.applications.VGG19, 224), "mobilenet": (keras.applications.MobileNet, 224),
        "resnet50": (keras.applications.ResNet50, 224), "inception": (keras.applications.InceptionV3, 299)}
EPOCHS = {"vgg19": 20, "mobilenet": 100, "resnet50": 100, "inception": 50}
STAGE2_EPOCHS, CUT, BATCH, K = 50, 249, 50, 3


def subset(a, n):
    """n evenly spaced rows, so every class appears (rows are ordered by class)."""
    return a if not n else a[np.linspace(0, len(a) - 1, n).astype(int)]


def extract(base, fdir, limit=None):
    meta_path = os.path.join(fdir, base + "_meta.json")
    if os.path.exists(meta_path):
        print(f"{base}: features already extracted ({meta_path})")
        return
    os.makedirs(fdir, exist_ok=True)
    app, size = APPS[base]
    net = app(weights="imagenet", include_top=False, input_shape=(size, size, 3))
    outputs = [keras.layers.GlobalAveragePooling2D()(net.output)]
    if base == "inception":
        outputs.append(net.layers[CUT - 1].output)  # mixed8 (8x8x1280), input of the fine-tuned blocks
    extractor = keras.Model(net.input, outputs)
    xtr, _, xdv, _, xte, _, _ = load(K, "uint8")
    t0 = time.time()
    pooled = {}
    for split, x in (("train", xtr), ("dev", xdv), ("test", xte)):
        x = subset(x, limit and (limit if split == "train" else limit // 3))
        mix = None
        if base == "inception":  # float16 memory map: about 9.8 GB for all 60,000 images
            mix = np.lib.format.open_memmap(os.path.join(fdir, f"mixed8_{split}.npy"), mode="w+",
                                            dtype=np.float16, shape=(len(x), 8, 8, 1280))
        parts, i = [], 0
        for b in resized_batches(x, size, BATCH):  # batch 100 ran out of the 8 GB GPU
            o = extractor.predict_on_batch(b)  # inference mode: the network stays frozen
            parts.append(o[0] if mix is not None else o)
            if mix is not None:
                mix[i:i + len(b)] = o[1].astype(np.float16)
            i += len(b)
        if mix is not None:
            mix.flush()
        pooled[split] = np.concatenate(parts)
        print(f"{base} {split}: {pooled[split].shape}")
    np.savez(os.path.join(fdir, base + ".npz"), f_train=pooled["train"], f_dev=pooled["dev"], f_test=pooled["test"])
    with open(meta_path, "w") as f:  # written last: it marks a complete extraction
        json.dump({"extract_time_s": round(time.time() - t0, 1), "dim": int(pooled["train"].shape[1])}, f)
    print(f"{base}: extraction done in {time.time() - t0:.0f}s")


class Feats(keras.utils.PyDataset):
    """Batches from a (memory-mapped) feature array, converted to float32 per batch."""

    def __init__(self, x, y, shuffle):
        super().__init__()
        self.x, self.y, self.shuffle = x, y, shuffle
        self.order = np.arange(len(x))
        self.on_epoch_end()

    def __len__(self):
        return -(-len(self.x) // BATCH)

    def __getitem__(self, i):
        idx = np.sort(self.order[i * BATCH:(i + 1) * BATCH])  # sorted reads are faster on a memmap
        return np.asarray(self.x[idx], dtype=np.float32), self.y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


def merge(*hists):
    h = {}
    for x in hists:
        for k, v in x.history.items():
            h.setdefault(k, []).extend(v)
    return h


def predict(model, x):
    return np.concatenate([model.predict_on_batch(b) for b, _ in Feats(x, np.zeros(len(x)), False)]).argmax(1)


def train(base, fdir, epochs=None, stage2_epochs=None, limit=None):
    meta = json.load(open(os.path.join(fdir, base + "_meta.json")))
    f = np.load(os.path.join(fdir, base + ".npz"))
    _, ytr, _, ydv, _, yte, names = load(K, "uint8")
    ytr, ydv, yte = subset(ytr, limit), subset(ydv, limit and limit // 3), subset(yte, limit and limit // 3)
    e1 = EPOCHS[base] if epochs is None else epochs
    e2 = (STAGE2_EPOCHS if stage2_epochs is None else stage2_epochs) if base == "inception" else 0
    params = {"stage1": {"epochs": e1, "optimizer": "adam(clipnorm=1)", "batch_size": BATCH},
              "stage2": {"epochs": e2, "optimizer": "sgd(lr=1e-4, momentum=0.9, clipnorm=1)",
                         "first_trainable_layer": CUT} if e2 else None,
              "option": "A (precomputed features)", "extract_time_s": meta["extract_time_s"]}
    with Run("TL", base, K, "resized", params, hardware="GPU", seed=7) as r:
        keras.utils.set_random_seed(7)
        head = keras.Sequential([keras.Input((meta["dim"],)), layers.Dense(128, activation="relu"),
                                 layers.Dense(K, activation="softmax")])
        head.compile(keras.optimizers.Adam(clipnorm=1), "sparse_categorical_crossentropy", metrics=["accuracy"])
        hists = []
        with r.fit_timer():
            hists.append(head.fit(Feats(f["f_train"], ytr, True), epochs=e1, verbose=2,
                                  validation_data=Feats(f["f_dev"], ydv, False), callbacks=[r.checkpoint_cb()]))
        final, dev_in, test_in = head, f["f_dev"], f["f_test"]
        if e2:  # InceptionV3: fine-tune layers 249+ with the head, starting from mixed8
            app, size = APPS[base]
            net = app(weights="imagenet", include_top=False, input_shape=(size, size, 3))
            top = keras.Model(net.layers[CUT - 1].output, net.output)
            top.trainable = True
            inp = keras.Input((8, 8, 1280))
            final = keras.Model(inp, head(layers.GlobalAveragePooling2D()(top(inp))))
            final.compile(keras.optimizers.SGD(1e-4, momentum=0.9, clipnorm=1), "sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            m8 = {s: np.load(os.path.join(fdir, f"mixed8_{s}.npy"), mmap_mode="r") for s in ("train", "dev", "test")}
            ckpt2 = keras.callbacks.BackupAndRestore(os.path.join(r.ckpt_dir, "stage2"), delete_checkpoint=False)
            with r.fit_timer():
                hists.append(final.fit(Feats(m8["train"], ytr, True), epochs=e2, verbose=2,
                                       validation_data=Feats(m8["dev"], ydv, False), callbacks=[ckpt2]))
            dev_in, test_in = m8["dev"], m8["test"]
        h = merge(*hists)
        r.history_csv(h)
        r.curve_png(h)
        r.params["head_sec_per_epoch"] = round(r.fit_time / (e1 + e2), 2)
        # validation accuracy is the reference's metric; the true test accuracy is recorded too
        r.finish(ydv, predict(final, dev_in), names, yte, predict(final, test_in))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", choices=list(APPS), required=True)
    ap.add_argument("--epochs", type=int, help="head epochs (default: vgg19 20, mobilenet/resnet50 100, inception 50)")
    ap.add_argument("--stage2-epochs", type=int, help="InceptionV3 fine-tuning epochs (default 50)")
    ap.add_argument("--limit", type=int, help="use only this many training images (quick test)")
    add_out_arg(ap)
    a = ap.parse_args()
    set_out(a.out)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    fdir = out("features") if a.out else os.path.join(ROOT, "data", "features")
    if a.limit:
        fdir = os.path.join(fdir, f"limit{a.limit}")
    extract(a.base, fdir, a.limit)
    train(a.base, fdir, a.epochs, a.stage2_epochs, a.limit)

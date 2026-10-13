"""CNN V1-V4 on raw or binarized 28x28 images.

V1: conv-pool-conv-pool-Dense(128). V2: one conv-pool. V3: V2 without pooling. V4: V2 with Dense(64).
Raw 0-255 input (no normalisation), Adam, batch 500, 100 epochs.

    python src/train_cnn.py --version V1 --input raw --classes 50
"""
import argparse
import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import tensorflow as tf
from keras import layers

from common import Run, add_out_arg, set_out
from data import load

CHECK_EPOCH = 5
MAX_SEEDS = 5


def build(version, k):
    L = [keras.Input((28, 28, 1)), layers.Conv2D(14, 3, activation="relu")]
    if version != "V3":
        L.append(layers.MaxPool2D(2))
    if version == "V1":
        L += [layers.Conv2D(14, 3, activation="relu"), layers.MaxPool2D(2)]
    L += [layers.Flatten(), layers.Dense(64 if version == "V4" else 128, activation="relu"),
          layers.Dense(k, activation="softmax")]
    return keras.Sequential(L)


class CollapseCheck(keras.callbacks.Callback):
    """Stop if validation accuracy is still near chance (1.5/K) after CHECK_EPOCH epochs."""

    def __init__(self, k):
        super().__init__()
        self.limit, self.collapsed = 1.5 / k, False

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == CHECK_EPOCH and logs["val_accuracy"] < self.limit:
            self.collapsed = True
            self.model.stop_training = True


def train(version, variant, k, seed, epochs):
    """One training run; returns False if it collapsed."""
    params = {"epochs": epochs, "batch_size": 500, "optimizer": "adam", "lr": 0.001, "seed": seed}
    with Run("CNN", version, k, variant, params, hardware="GPU", seed=seed) as r:
        keras.utils.set_random_seed(seed)
        xtr, ytr, xdv, ydv, xte, yte, names = load(k, "image" if variant == "raw" else "binarized_image")
        model = build(version, k)
        # sparse labels: same loss as categorical cross-entropy on one-hot labels
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        print(f"params={model.count_params()}")
        check = CollapseCheck(k)
        with r.fit_timer():
            hist = model.fit(xtr, ytr, batch_size=500, epochs=epochs, validation_data=(xdv, ydv),
                             verbose=2, callbacks=[r.checkpoint_cb(), check])
        if check.collapsed:
            print(f"[{r.id}] collapsed: dev acc below {check.limit:.3f} after {CHECK_EPOCH} epochs")
            shutil.rmtree(r.ckpt_dir, ignore_errors=True)  # a retry must start fresh, not resume
            return False
        r.history_csv(hist)
        r.curve_png(hist)
        pred_dev = model.predict(xdv, batch_size=500, verbose=0).argmax(1)
        final = version == "V2" and variant == "raw" and k == 50  # the only CNN scored on test
        pred_te = model.predict(xte, batch_size=500, verbose=0).argmax(1) if final else None
        r.finish(ydv, pred_dev, names, yte if final else None, pred_te)
        return True


def main(version, variant, k, seed=7, epochs=100):
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    for s in range(seed, seed + MAX_SEEDS):  # unnormalised input can collapse: retry with the next seed
        if train(version, variant, k, s, epochs):
            return
    raise SystemExit(f"collapsed with seeds {seed}..{seed + MAX_SEEDS - 1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", choices=["V1", "V2", "V3", "V4"], required=True)
    ap.add_argument("--input", choices=["raw", "binarized"], required=True)
    ap.add_argument("--classes", type=int, choices=[3, 10, 50], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=100)
    add_out_arg(ap)
    a = ap.parse_args()
    set_out(a.out)
    main(a.version, a.input, a.classes, a.seed, a.epochs)

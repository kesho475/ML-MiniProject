"""Phase 6B: CNN V1-V4 (report Tables 3-4, Fig. 3).
Usage: python src/cnn.py <V1..V4> <raw|binarized> <k> [seed=7] [--epochs N] [--smoke]

Spec: raw 0-255 input (no normalization), Adam default, cross-entropy, batch 500, 100 epochs.
Exit code 3 = collapse: dev accuracy still at chance after 5 epochs (Q3); the launcher reruns with seed+1.
--smoke: model name gets "-smoke" so it never counts as a real result.
"""
import os
import shutil
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import tensorflow as tf
from keras import layers

from data import load
from harness import Run

COLLAPSE_EXIT = 3
CHECK_EPOCH = 5


def build(version, k):
    """V1: conv-pool-conv-pool-D128. V2: one conv-pool. V3: V2 without pool. V4: V2 with D64 (Fig. 3)."""
    L = [keras.Input((28, 28, 1)), layers.Conv2D(14, 3, activation="relu")]
    if version != "V3":
        L.append(layers.MaxPool2D(2))
    if version == "V1":
        L += [layers.Conv2D(14, 3, activation="relu"), layers.MaxPool2D(2)]
    L += [layers.Flatten(), layers.Dense(64 if version == "V4" else 128, activation="relu"),
          layers.Dense(k, activation="softmax")]
    return keras.Sequential(L)


class CollapseCheck(keras.callbacks.Callback):
    """Stop if dev accuracy is still near chance (1/K) after CHECK_EPOCH epochs."""

    def __init__(self, k):
        super().__init__()
        self.limit, self.collapsed = 1.5 / k, False

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == CHECK_EPOCH and logs["val_accuracy"] < self.limit:
            self.collapsed = True
            self.model.stop_training = True


def main(version, variant, k, seed=7, epochs=100, smoke=False):
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    params = {"epochs": epochs, "batch_size": 500, "optimizer": "adam", "lr": 0.001, "seed": seed}
    with Run("CNN", version + ("-smoke" if smoke else ""), k, variant, params, hardware="GPU", seed=seed) as r:
        keras.utils.set_random_seed(seed)
        xtr, ytr, xdv, ydv, xte, yte, names = load(k, "image" if variant == "raw" else "binarized_image")
        model = build(version, k)
        # sparse labels = same loss as categorical cross-entropy on one-hot labels
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        print(f"params={model.count_params()}")
        check = CollapseCheck(k)
        with r.fit_timer():
            hist = model.fit(xtr, ytr, batch_size=500, epochs=epochs, validation_data=(xdv, ydv),
                             verbose=2, callbacks=[r.checkpoint_cb(), check])
        if check.collapsed:
            print(f"[{r.id}] COLLAPSED: dev acc below {check.limit:.3f} after {CHECK_EPOCH} epochs")
            shutil.rmtree(r.ckpt_dir, ignore_errors=True)  # a retry of this id must start fresh, not resume
            sys.exit(COLLAPSE_EXIT)
        r.history_csv(hist)
        r.curve_png(hist)
        pred_dev = model.predict(xdv, batch_size=500, verbose=0).argmax(1)
        final = version == "V2" and variant == "raw" and k == 50  # only the final check touches test
        pred_te = model.predict(xte, batch_size=500, verbose=0).argmax(1) if final else None
        r.finish(ydv, pred_dev, names, yte if final else None, pred_te)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    ep = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 100
    if "--epochs" in sys.argv:
        a.remove(str(ep))
    main(a[0], a[1], int(a[2]), int(a[3]) if len(a) > 3 else 7, ep, "--smoke" in sys.argv)

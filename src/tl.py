"""Phase 7 (Option A, step 2): train the head on the saved features.
Usage: python src/tl.py <vgg19|mobilenet|resnet50|inception> [--smoke]   (run tl_features.py first)

Head: Dense(128, relu) -> Dense(3, softmax) on globally pooled features. Stage 1: Adam(clipnorm=1), batch 50.
Inception stage 2: layers 249+ of the base (top 2 blocks) + head, trained on the saved `mixed8` features with
SGD(lr=1e-4, momentum=0.9, clipnorm=1). Epochs: VGG19 20, MobileNet 100, ResNet50 100, Inception 50 + 50.
Accuracy is dev accuracy (as in the reference); true test accuracy is also recorded. Times are head-only (feature extraction time
is in params.extract_time_s), so they are NOT comparable with the report's full-network seconds per epoch.
--smoke: reads data/features/smoke/, 2 epochs per stage, model name gets "-smoke".
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from data import load
from harness import Run
from tl_features import APPS, SMOKE_N, feat_dir

EPOCHS = {"vgg19": 20, "mobilenet": 100, "resnet50": 100, "inception": 50}
STAGE2_EPOCHS, CUT, BATCH = 50, 249, 50
K = 3


class Feats(keras.utils.PyDataset):
    """Batches from a (memory-mapped) float16/32 feature array, converted to float32 per batch."""

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
    """Concatenate Keras histories (stage 1 then stage 2) for the CSV and curve."""
    out = {}
    for h in hists:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(v)
    return out


def predict(model, x):
    return np.concatenate([model.predict_on_batch(b) for b, _ in Feats(x, np.zeros(len(x)), False)]).argmax(1)


def main(base, smoke=False):
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    fdir = feat_dir(smoke)
    meta = json.load(open(os.path.join(fdir, base + "_meta.json")))
    f = np.load(os.path.join(fdir, base + ".npz"))
    _, ytr, _, ydv, _, yte, names = load(K, "uint8")
    if smoke:  # same strided subset as tl_features.py --smoke
        pick = lambda y, s: y[np.linspace(0, len(y) - 1, SMOKE_N[s]).astype(int)]
        ytr, ydv, yte = pick(ytr, "train"), pick(ydv, "dev"), pick(yte, "test")
    e1 = 2 if smoke else EPOCHS[base]
    e2 = (1 if smoke else STAGE2_EPOCHS) if base == "inception" else 0
    model_name = base + ("-smoke" if smoke else "")
    params = {"stage1": {"epochs": e1, "optimizer": "adam(clipnorm=1)", "batch_size": BATCH},
              "stage2": {"epochs": e2, "optimizer": "sgd(lr=1e-4, momentum=0.9, clipnorm=1)",
                         "first_trainable_layer": CUT} if e2 else None,
              "option": "A (precomputed features)", "extract_time_s": meta["extract_time_s"]}
    with Run("TL", model_name, K, "resized", params, hardware="GPU", seed=7) as r:
        keras.utils.set_random_seed(7)
        head = keras.Sequential([keras.Input((meta["dim"],)), layers.Dense(128, activation="relu"),
                                 layers.Dense(K, activation="softmax")])
        head.compile(keras.optimizers.Adam(clipnorm=1), "sparse_categorical_crossentropy", metrics=["accuracy"])
        app, size = APPS[base]
        print(f"params base+head={app(weights=None, include_top=False, input_shape=(size, size, 3)).count_params() + head.count_params()}")
        hists = []
        with r.fit_timer():
            hists.append(head.fit(Feats(f["f_train"], ytr, True), epochs=e1, verbose=2,
                                  validation_data=Feats(f["f_dev"], ydv, False), callbacks=[r.checkpoint_cb()]))
        final, dev_in, test_in = head, f["f_dev"], f["f_test"]
        if e2:
            net = app(weights="imagenet", include_top=False, input_shape=(size, size, 3))
            top = keras.Model(net.layers[CUT - 1].output, net.output)  # layers 249+, input = mixed8
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
        n_ep = e1 + e2
        print(f"head-only seconds/epoch={r.fit_time / n_ep:.1f} over {n_ep} epochs")
        r.params["head_sec_per_epoch"] = round(r.fit_time / n_ep, 2)
        r.finish(ydv, predict(final, dev_in), names, yte, predict(final, test_in))


if __name__ == "__main__":
    main(sys.argv[1], "--smoke" in sys.argv)

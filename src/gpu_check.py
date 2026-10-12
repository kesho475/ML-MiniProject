# Phase 1 check: TF sees the GPU, and a tiny fit really runs on it.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)  # don't grab all 8 GB
m = tf.keras.Sequential([tf.keras.Input((20,)), tf.keras.layers.Dense(8, activation="relu"), tf.keras.layers.Dense(3, activation="softmax")])
m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
with tf.device("/GPU:0"):
    h = m.fit(np.random.rand(512, 20), np.random.randint(0, 3, 512), epochs=2, verbose=2)
print("fit ok, loss", h.history["loss"][-1])

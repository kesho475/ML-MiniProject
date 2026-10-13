# Drawing: A New Way To Search — Reproduce & Extend

We reproduce the experiments of the report *Drawing: A New Way To Search*, which classifies hand-drawn doodles
so that drawing could be used as a search interface. We rerun all of its models (logistic regression, SVM,
four CNN variants and transfer learning) and add one new model family, XGBoost.

## Dataset

[Google QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset), numpy bitmap version: 28×28 grayscale
drawings. As in the report we use 3, 10 and 50 classes with 20,000 drawings per class, split per class
80 / 10 / 10 into train / validation / test (16,000 / 2,000 / 2,000). The split is built once and shared by
every model.

## Install

Python 3.12 and, for the CNN, transfer learning and XGBoost runs, an NVIDIA GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If TensorFlow does not see the GPU, add the pip-installed CUDA libraries to `LD_LIBRARY_PATH`
(`.venv/lib/python3.12/site-packages/nvidia/*/lib`).

## Get the data

```bash
python src/download.py     # 50 QuickDraw .npy files into data/raw/ (about 5.6 GB)
python src/data.py         # fixed split: data/processed/qd_3.npz, qd_10.npz, qd_50.npz
python src/data.py --70k   # optional: 70,000 drawings per class for the SVM data-size check
```

## Run the models

Every run appends one row to `results/all_results.csv`; finished runs are skipped when a queue is rerun.

| Family | Command |
| --- | --- |
| Logistic regression | `python src/lr.py` (3, 10 and 50 classes) |
| SVM | `bash src/launch_svm.sh` (4 kernels × 3 and 10 classes), or `python src/svm.py <linear\|rbf\|poly\|sigmoid> <k>` |
| SVM, 70k per class | `bash src/launch_svm70k.sh` |
| CNN | `bash src/launch_cnn.sh`, or `python src/cnn.py <V1..V4> <raw\|binarized> <k>` |
| Transfer learning | `bash src/launch_tl.sh`, or `python src/tl_features.py <base>` then `python src/tl.py <base>` (base: vgg19, mobilenet, resnet50, inception) |
| XGBoost, tuned (time-capped) | `bash src/launch_xgb.sh`, or `python src/xgb.py <stage1\|stage2\|stage3>` |
| XGBoost, extended (no cap, wider search) | `bash src/launch_xgb_ext.sh` (needs the tuned XGBoost results first) |

`python src/compare.py [LR|SVM|CNN|TL|XGB|svm70k]` prints reported vs our numbers in the terminal.

## Results and slides

```bash
python src/report.py        # rebuilds results/tables/*.csv and results/figures/*.png
python src/make_slides.py   # rebuilds presentation/ML_MiniProject.pptx
```

- `results/tables/`: reported vs ours per model family, the SVM data-size check, all our models side by side,
  and XGBoost before / after the extended run. Each table explains every gap in plain words.
- `results/figures/`: the figures used in the slides and the key comparison charts.
- `results/xgb_improvement.md`: what the extended XGBoost run changed.
- `results/all_results.csv` (our runs) and `results/reported.csv` (the report's numbers): the data every table
  is built from.
- `presentation/`: the slides (`ML_MiniProject.pptx` and `.pdf`).

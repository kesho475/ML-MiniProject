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
python src/data.py download       # 50 QuickDraw .npy files into data/raw/ (about 5.6 GB)
python src/data.py split          # fixed split: data/processed/qd_3.npz, qd_10.npz, qd_50.npz
python src/data.py split --70k    # optional: 70,000 drawings per class for the SVM data-size check
```

## Train the models

Each script trains one configuration and appends one row to `results/all_results.csv`. Run any script with
`--help` for its options; `--out <folder>` writes everything to another folder instead of `results/`.

| Model | Example |
| --- | --- |
| Logistic regression | `python src/train_lr.py --classes 3 10 50` |
| SVM (linear, rbf, poly, sigmoid; 3 or 10 classes) | `python src/train_svm.py --kernel rbf --classes 3` |
| SVM with 70,000 drawings per class | `python src/train_svm.py --kernel rbf --classes 3 --70k` |
| CNN (V1–V4, raw or binarized) | `python src/train_cnn.py --version V1 --input raw --classes 50` |
| Transfer learning (vgg19, mobilenet, resnet50, inception; 3 classes) | `python src/train_transfer.py --base vgg19` |
| XGBoost: Optuna search | `python src/train_xgboost.py search --classes 10` |
| XGBoost: fully trained run of the best setting | `python src/train_xgboost.py train --classes 50 --study xgb_10` |

## Results and slides

- `results/tables/`: reported vs ours per model family, the SVM data-size check, all our models side by side,
  and XGBoost before / after the extended run. Each table explains every gap in plain words.
- `results/figures/`: the figures used in the slides and the key comparison charts.
- `results/xgb_improvement.md`: what the extended XGBoost run changed.
- `results/all_results.csv` (our runs) and `results/reported.csv` (the report's numbers): the data every table
  was built from.
- `presentation/`: the slides (`ML_MiniProject.pptx` and `.pdf`).

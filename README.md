# PCL Detection: Patronizing and Condescending Language Classification

NLP project for detecting patronizing and condescending language (PCL) towards vulnerable communities using binary classification. Based on **SemEval 2022 Task 4, Subtask 1** — the _Don't Patronize Me!_ dataset.

Coursework for Imperial Computing 60035 Natural Language Processing.

---

## Repository Structure

```text
pcl-detection/
├── data/
│   ├── raw/                    # Original SemEval dataset files
│   └── processed/              # train.csv / dev.csv / test.csv
├── scripts/
│   ├── process_data.py         # Converts raw TSV files → processed CSVs
│   └── verify_data.py          # Sanity-checks processed splits
├── BestModel/                  # RoBERTa-base + multi-task + focal loss
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── requirements.txt
├── dev.txt                     # Submission predictions for dev set
├── test.txt                    # Submission predictions for test set
└── report/                     # LaTeX report
```

---

## Setup

### Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with **Python 3.12, PyTorch 2.1, CUDA 12.1** on an RTX 4080 (16 GB VRAM).

### Jupyter Notebook

Register the venv as a Jupyter kernel so `notebooks/eda.ipynb` uses the installed packages:

```bash
python -m ipykernel install --user --name=pcl-venv --display-name "PCL (venv)"
jupyter notebook notebooks/eda.ipynb
```

Select **PCL (venv)** as the kernel when the notebook opens.

### Data

The processed splits (`data/processed/`) and raw files (`data/raw/`) are already prepared in the repo — no data preparation is needed to run the model or notebook.

> **Regenerating from scratch:** If you need to rebuild the processed CSVs, ensure the four raw SemEval files are present in `data/raw/` (`dontpatronizeme_pcl.tsv`, `train_semeval_parids-labels.csv`, `dev_semeval_parids-labels.csv`, `task4_test.tsv`), then run:
>
> ```bash
> python scripts/process_data.py   # produces data/processed/{train,dev,test}.csv
> python scripts/verify_data.py    # optional sanity check
> ```

---

## Model — RoBERTa-base

Multi-task RoBERTa with keyword injection, focal loss, and threshold optimisation.

| Component | Detail                                   |
| --------- | ---------------------------------------- |
| Backbone  | `FacebookAI/roberta-base`                |
| Input     | `[{keyword}] {text}`                     |
| Loss      | Focal loss (γ=2) + auxiliary 7-label BCE |
| Imbalance | WeightedRandomSampler (33% positive)     |
| Threshold | Dev-set sweep (0.05–0.95, step 0.01)     |

**Train:**

```bash
cd BestModel
python train.py
```

Saves best checkpoint to `BestModel/checkpoints/best_model.pt` and optimal threshold to `BestModel/checkpoints/best_threshold.txt`.

**Predict (generates `dev.txt` and `test.txt`):**

```bash
python predict.py
```

---

## Submission Files

`dev.txt` and `test.txt` in the project root are the final predictions, one label per line (`0` or `1`):

- `dev.txt` — 2094 lines (dev set)
- `test.txt` — 3832 lines (test set)

---

## Reproducibility

All runs use `--seed 42` by default. Predictions are fully reproducible from `best_model.pt` alone, as the optimal threshold is stored alongside it.

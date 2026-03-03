"""
error_analysis.py  —  Local evaluation and error analysis for the BestModel.

Outputs
-------
  BestModel/analysis/metrics.txt         global metrics + confusion matrix
  BestModel/analysis/fp_examples.csv     false positives with probabilities
  BestModel/analysis/fn_examples.csv     false negatives with probabilities
  BestModel/analysis/keyword_errors.csv  per-keyword TP/FP/FN/TN counts
  BestModel/analysis/pr_curve.png        precision-recall curve
  BestModel/analysis/error_summary.txt   human-readable summary for report
"""

import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_CSV     = os.path.join(ROOT, "data/processed/dev.csv")
DEV_TXT     = os.path.join(ROOT, "dev.txt")
PROBS_NP    = os.path.join(ROOT, "BestModel/checkpoints/best_dev_probs.npy")
OUT_DIR     = os.path.join(ROOT, "BestModel/analysis")
FIGURES_DIR = os.path.join(ROOT, "report/figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CAT_NAMES = [
    "Unbalanced Power Relations",
    "Shallow Solution",
    "Presupposition",
    "Authority Voice",
    "Metaphor",
    "Compassion",
    "The Poorer, the Merrier",
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
dev = pd.read_csv(DEV_CSV)
preds = np.loadtxt(DEV_TXT, dtype=int)
probs = np.load(PROBS_NP)

def parse_label(s):
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return [0] * 7

labels_7 = dev["label"].apply(parse_label).tolist()
binary_true = np.array([int(sum(l) > 0) for l in labels_7])

dev["true"]    = binary_true
dev["pred"]    = preds
dev["prob"]    = probs
dev["tok_len"] = dev["text"].str.split().str.len()
dev["labels_7"] = labels_7

# ---------------------------------------------------------------------------
# Confusion matrix quadrants
# ---------------------------------------------------------------------------
tp_df = dev[(dev["true"] == 1) & (dev["pred"] == 1)].copy()
fp_df = dev[(dev["true"] == 0) & (dev["pred"] == 1)].copy()
fn_df = dev[(dev["true"] == 1) & (dev["pred"] == 0)].copy()
tn_df = dev[(dev["true"] == 0) & (dev["pred"] == 0)].copy()

cm = confusion_matrix(binary_true, preds)
TN, FP, FN, TP = cm.ravel()

# ---------------------------------------------------------------------------
# 1. Global metrics
# ---------------------------------------------------------------------------
report = classification_report(
    binary_true, preds, target_names=["Non-PCL", "PCL"], digits=4
)
ap = average_precision_score(binary_true, probs)
fpr = FP / (FP + TN)
fnr = FN / (FN + TP)

metrics_txt = f"""=== Global Metrics ===
{report}
Confusion Matrix:
         Predicted Non-PCL   Predicted PCL
True Non-PCL     {TN:>6}          {FP:>6}
True PCL         {FN:>6}          {TP:>6}

False Positive Rate (FPR): {fpr:.4f}   ({FP}/{FP+TN})
False Negative Rate (FNR): {fnr:.4f}   ({FN}/{FN+TP})
Average Precision (PR-AUC): {ap:.4f}

Error counts: TP={TP}  FP={FP}  FN={FN}  TN={TN}

=== Token Length by Quadrant ===
"""
for name, df in [("TP", tp_df), ("FP", fp_df), ("FN", fn_df), ("TN", tn_df)]:
    metrics_txt += (
        f"  {name}: mean={df['tok_len'].mean():.1f}  "
        f"median={df['tok_len'].median():.0f}  "
        f"max={df['tok_len'].max()}\n"
    )

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(metrics_txt)
print(metrics_txt)

# ---------------------------------------------------------------------------
# 2. Per-keyword error breakdown
# ---------------------------------------------------------------------------
kw_rows = []
for kw, grp in dev.groupby("keyword"):
    t = grp["true"].values
    p = grp["pred"].values
    kw_rows.append({
        "keyword": kw,
        "n_total":  len(grp),
        "n_pcl":    int(t.sum()),
        "TP": int(((t==1)&(p==1)).sum()),
        "FP": int(((t==0)&(p==1)).sum()),
        "FN": int(((t==1)&(p==0)).sum()),
        "TN": int(((t==0)&(p==0)).sum()),
        "recall_pcl": round(int(((t==1)&(p==1)).sum()) / max(int(t.sum()), 1), 3),
        "precision_pcl": round(
            int(((t==1)&(p==1)).sum()) /
            max(int(((p==1)).sum()), 1), 3
        ),
    })
kw_df = pd.DataFrame(kw_rows).sort_values("FN", ascending=False)
kw_df.to_csv(os.path.join(OUT_DIR, "keyword_errors.csv"), index=False)
print("=== Per-keyword breakdown (sorted by FN) ===")
print(kw_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. PCL category breakdown among false negatives
# ---------------------------------------------------------------------------
fn_cats = np.array([labels_7[i] for i in fn_df.index])
fn_cat_counts = fn_cats.sum(axis=0).astype(int) if len(fn_cats) else np.zeros(7, int)

tp_cats = np.array([labels_7[i] for i in tp_df.index])
tp_cat_counts = tp_cats.sum(axis=0).astype(int) if len(tp_cats) else np.zeros(7, int)

print("\n=== PCL Category Breakdown: FN vs TP ===")
cat_summary = []
for i, cat in enumerate(CAT_NAMES):
    total = fn_cat_counts[i] + tp_cat_counts[i]
    miss_rate = fn_cat_counts[i] / max(total, 1)
    cat_summary.append({
        "category": cat,
        "TP": tp_cat_counts[i],
        "FN": fn_cat_counts[i],
        "total_positives": total,
        "miss_rate": round(miss_rate, 3),
    })
    print(f"  {cat:<22}  TP={tp_cat_counts[i]:>3}  FN={fn_cat_counts[i]:>3}  "
          f"miss_rate={miss_rate:.3f}")

cat_df = pd.DataFrame(cat_summary)
cat_df.to_csv(os.path.join(OUT_DIR, "category_errors.csv"), index=False)

# ---------------------------------------------------------------------------
# 4. High-confidence errors
# ---------------------------------------------------------------------------
fp_top = fp_df.nlargest(10, "prob")[["keyword", "prob", "text"]].copy()
fn_top = fn_df.nsmallest(10, "prob")[["keyword", "prob", "text"]].copy()

fp_top.to_csv(os.path.join(OUT_DIR, "fp_examples.csv"), index=False)
fn_top.to_csv(os.path.join(OUT_DIR, "fn_examples.csv"), index=False)

print("\n=== Top-5 Confident False Positives ===")
for _, row in fp_top.head(5).iterrows():
    print(f"  [{row['keyword']}] p={row['prob']:.3f}  {str(row['text'])[:120]}")

print("\n=== Top-5 Confident False Negatives ===")
for _, row in fn_top.head(5).iterrows():
    print(f"  [{row['keyword']}] p={row['prob']:.3f}  {str(row['text'])[:120]}")

# ---------------------------------------------------------------------------
# 5. Precision-Recall curve
# ---------------------------------------------------------------------------
prec_curve, rec_curve, thresholds = precision_recall_curve(binary_true, probs)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rec_curve, prec_curve, lw=1.8, color="#2563EB", label=f"PR curve (AP={ap:.3f})")
ax.axhline(binary_true.mean(), ls="--", color="gray", lw=1, label="Random baseline")
rec_op  = TP / max(TP + FN, 1)
prec_op = TP / max(TP + FP, 1)
ax.scatter([rec_op], [prec_op], color="red", zorder=5, label="Operating point (t=0.50)")

ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curve — BestModel (dev set)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "pr_curve.png"), dpi=150)
print(f"\nPR curve saved.")

# ---------------------------------------------------------------------------
# 6. Probability distribution by class
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
ax2.hist(probs[binary_true==0], bins=40, alpha=0.6, color="#64748B", label="Non-PCL (true)")
ax2.hist(probs[binary_true==1], bins=40, alpha=0.6, color="#2563EB", label="PCL (true)")
ax2.axvline(0.50, color="red", ls="--", lw=1.5, label="Threshold (0.50)")
ax2.set_xlabel("Predicted probability", fontsize=11)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_title("Score Distribution by True Label", fontsize=12)
ax2.legend(fontsize=9)
plt.tight_layout()
fig2.savefig(os.path.join(FIGURES_DIR, "score_distribution.png"), dpi=150)
print("Score distribution saved.")

# ---------------------------------------------------------------------------
# 7. Human-readable summary
# ---------------------------------------------------------------------------
summary = f"""=== Error Analysis Summary ===

Model: RoBERTa-base dual-head (BestModel)
Dev set: {len(dev)} samples  |  PCL positives: {int(binary_true.sum())} ({100*binary_true.mean():.1f}%)
Decision threshold: 0.50

Global performance
  F1 (PCL):        {f1_score(binary_true, preds):.4f}
  Precision (PCL): {precision_score(binary_true, preds, zero_division=0):.4f}
  Recall (PCL):    {recall_score(binary_true, preds, zero_division=0):.4f}
  PR-AUC:          {ap:.4f}
  FPR:             {fpr:.4f}
  FNR:             {fnr:.4f}

Confusion matrix: TP={TP}  FP={FP}  FN={FN}  TN={TN}

Most-missed PCL category (highest miss rate):
{cat_df.sort_values('miss_rate', ascending=False).iloc[0]['category']}  miss_rate={cat_df.sort_values('miss_rate', ascending=False).iloc[0]['miss_rate']}

Keyword with most false negatives:
{kw_df.iloc[0]['keyword']}  FN={kw_df.iloc[0]['FN']}  recall={kw_df.iloc[0]['recall_pcl']}

Token length:
  FP mean: {fp_df['tok_len'].mean():.1f}  FN mean: {fn_df['tok_len'].mean():.1f}
  TP mean: {tp_df['tok_len'].mean():.1f}  TN mean: {tn_df['tok_len'].mean():.1f}
"""
with open(os.path.join(OUT_DIR, "error_summary.txt"), "w") as f:
    f.write(summary)
print(summary)
print(f"\nAll outputs written to {OUT_DIR}/")

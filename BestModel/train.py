import argparse
import ast
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import PCLDataset
from model import FocalLoss, PCLModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# EDA augmentation — random word deletion on PCL positives
# ---------------------------------------------------------------------------
def augment_positives(
    df: pd.DataFrame,
    n_aug: int = 3,
    drop_prob: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Augment PCL-positive rows by randomly dropping words (EDA-RD).

    With n_aug=3 and 794 positives the positive class grows 794 → 3176,
    improving recall without introducing external data.
    """
    rng = random.Random(seed)

    def _is_positive(label_str) -> bool:
        try:
            return sum(ast.literal_eval(str(label_str))) > 0
        except Exception:
            return False

    pos_mask = df["label"].apply(_is_positive)
    pos_df = df[pos_mask]

    augmented_rows = []
    for _, row in pos_df.iterrows():
        tokens = str(row["text"]).split()
        for _ in range(n_aug):
            kept = [t for t in tokens if rng.random() > drop_prob]
            new_row = row.copy()
            new_row["text"] = " ".join(kept) if kept else str(row["text"])
            augmented_rows.append(new_row)

    if not augmented_rows:
        return df

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------
def build_weighted_sampler(binary_labels: torch.Tensor, pos_ratio: float = 0.33):
    n_pos = binary_labels.sum().item()
    n_neg = len(binary_labels) - n_pos
    w_pos = pos_ratio / max(n_pos, 1)
    w_neg = (1.0 - pos_ratio) / max(n_neg, 1)
    weights = torch.where(binary_labels == 1,
                          torch.tensor(w_pos), torch.tensor(w_neg))
    return WeightedRandomSampler(weights.double(), len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def threshold_sweep(probs: np.ndarray, labels: np.ndarray, step: float = 0.01):
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.95 + step, step):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh, best_f1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["binary_label"].numpy()

            binary_logits, _ = model(input_ids, attention_mask)
            probs = torch.sigmoid(binary_logits.float()).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= threshold).astype(int)
    f1   = f1_score(all_labels, preds, pos_label=1, zero_division=0)
    prec = precision_score(all_labels, preds, pos_label=1, zero_division=0)
    rec  = recall_score(all_labels, preds, pos_label=1, zero_division=0)
    return f1, prec, rec, all_probs, all_labels


# ---------------------------------------------------------------------------
# Optimizer param groups
# ---------------------------------------------------------------------------
def get_optimizer_params(model, encoder_lr, head_lr, weight_decay):
    """Two-group AdamW: encoder layers at encoder_lr, heads at head_lr.

    For RoBERTa-base the standard encoder_lr=2e-5 is well-validated.
    Bias / LayerNorm params are excluded from weight decay.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    encoder_decay, encoder_nodecay = [], []
    head_decay,    head_nodecay    = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_nodecay = any(nd in name for nd in no_decay)
        # Params whose name starts with the encoder module (roberta.*)
        is_encoder = name.startswith("encoder.")
        if is_encoder:
            (encoder_nodecay if is_nodecay else encoder_decay).append(param)
        else:
            (head_nodecay    if is_nodecay else head_decay).append(param)

    return [
        {"params": encoder_decay,    "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": encoder_nodecay,  "lr": encoder_lr, "weight_decay": 0.0},
        {"params": head_decay,       "lr": head_lr,    "weight_decay": weight_decay},
        {"params": head_nodecay,     "lr": head_lr,    "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Loading data …")
    train_df = pd.read_csv(args.train_path)
    dev_df   = pd.read_csv(args.dev_path)
    log.info(f"  Train: {len(train_df):,}  |  Dev: {len(dev_df):,}")

    if args.n_aug > 0:
        train_df = augment_positives(
            train_df, n_aug=args.n_aug,
            drop_prob=args.aug_drop_prob, seed=args.seed,
        )
        log.info(f"  After EDA (n_aug={args.n_aug}): {len(train_df):,} rows")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = PCLDataset(train_df, tokenizer, max_length=args.max_length)
    dev_ds   = PCLDataset(dev_df,   tokenizer, max_length=args.max_length)

    n_pcl = train_ds.binary_labels.sum().item()
    log.info(f"  Train PCL: {int(n_pcl):,}/{len(train_ds):,} ({100*n_pcl/len(train_ds):.1f}%)")

    sampler = build_weighted_sampler(train_ds.binary_labels, args.pos_ratio)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.eval_batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info(f"Loading model: {args.model_name} …")
    model = PCLModel(
        model_name=args.model_name,
        num_aux_labels=7,
        dropout=args.dropout,
    ).float().to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Params: {total_params:,}  |  Trainable: {trainable_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    focal_criterion = FocalLoss(gamma=args.focal_gamma)
    aux_criterion   = nn.BCEWithLogitsLoss()

    # ── Optimiser & schedule ──────────────────────────────────────────────────
    param_groups = get_optimizer_params(
        model, args.encoder_lr, args.head_lr, args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups, eps=1e-8)

    total_steps  = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    log.info(f"  Steps: {total_steps:,}  |  Warmup: {warmup_steps:,}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1, patience_counter = 0.0, 0
    history = []

    log.info("=" * 60)
    log.info("Starting training …")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_labels  = batch["binary_label"].float().to(device)
            multi_labels   = batch["multi_label"].to(device)

            binary_logits, aux_logits = model(input_ids, attention_mask)

            focal_loss = focal_criterion(binary_logits.float(), binary_labels)
            aux_loss   = aux_criterion(aux_logits.float(), multi_labels)
            loss = (focal_loss + args.lambda_aux * aux_loss) / args.grad_accum_steps

            loss.backward()
            total_loss += loss.item() * args.grad_accum_steps

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                log.info(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | "
                         f"Loss {total_loss/step:.4f}")

        f1, prec, rec, dev_probs, dev_labels = evaluate(model, dev_loader, device)
        avg_loss = total_loss / len(train_loader)
        log.info(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
                 f"dev_F1={f1:.4f} | dev_P={prec:.4f} | dev_R={rec:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss,
                        "dev_f1": f1, "dev_p": prec, "dev_r": rec})

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model_weights.pt"))
            np.save(os.path.join(args.output_dir, "best_dev_probs.npy"),  dev_probs)
            np.save(os.path.join(args.output_dir, "best_dev_labels.npy"), dev_labels)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "dev_f1": f1},
                       os.path.join(args.output_dir, "best_model.pt"))
            log.info(f"  *** New best dev F1 = {best_f1:.4f} — checkpoint saved ***")
        else:
            patience_counter += 1
            log.info(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                log.info(f"  Early stopping at epoch {epoch}.")
                break

    # ── Threshold sweep ───────────────────────────────────────────────────────
    log.info("\nRunning threshold sweep …")
    best_probs  = np.load(os.path.join(args.output_dir, "best_dev_probs.npy"))
    best_labels = np.load(os.path.join(args.output_dir, "best_dev_labels.npy"))
    best_thresh, best_thresh_f1 = threshold_sweep(best_probs, best_labels)
    log.info(f"  Optimal threshold = {best_thresh:.2f}  |  F1 = {best_thresh_f1:.4f}")

    with open(os.path.join(args.output_dir, "best_threshold.txt"), "w") as fh:
        fh.write(f"{best_thresh:.4f}\n")

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    log.info(f"\nDone. Best dev F1 = {best_f1:.4f}  "
             f"(threshold-adjusted: {best_thresh_f1:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train RoBERTa-base PCL detector")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Paths
    p.add_argument("--train_path",      default=os.path.join(root, "data/processed/train.csv"))
    p.add_argument("--dev_path",        default=os.path.join(root, "data/processed/dev.csv"))
    p.add_argument("--output_dir",      default=os.path.join(root, "BestModel/checkpoints"))
    # Model
    p.add_argument("--model_name",      default="roberta-base")
    p.add_argument("--max_length",      type=int,   default=256)
    p.add_argument("--dropout",         type=float, default=0.2)
    # Training
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--patience",        type=int,   default=5)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--eval_batch_size", type=int,   default=64)
    p.add_argument("--grad_accum_steps",type=int,   default=2,
                   help="Effective batch = 16 * 2 = 32")
    p.add_argument("--encoder_lr",      type=float, default=1.5e-5,
                   help="RoBERTa encoder LR — 1.5e-5 balances speed and stability")
    p.add_argument("--head_lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--warmup_ratio",    type=float, default=0.06)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    # Loss
    p.add_argument("--focal_gamma",     type=float, default=2.0)
    p.add_argument("--lambda_aux",      type=float, default=0.3)
    p.add_argument("--pos_ratio",       type=float, default=0.33)
    # Augmentation
    p.add_argument("--n_aug",           type=int,   default=0)
    p.add_argument("--aug_drop_prob",   type=float, default=0.10)
    # Misc
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--log_every",       type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

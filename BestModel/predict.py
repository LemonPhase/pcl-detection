"""
predict.py — Generate dev.txt and test.txt from the best RoBERTa checkpoint.

Usage
-----
    # Single model (default)
    python predict.py

    # Ensemble
    python predict.py --checkpoints checkpoints/best_model.pt path/to/run2/best_model.pt
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import PCLDataset
from model import PCLModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_probs(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, _ = model(input_ids, attention_mask)
            probs = torch.sigmoid(binary_logits.float()).cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.array(all_probs)


def load_model(checkpoint_path, model_name, device):
    model = PCLModel(model_name=model_name, num_aux_labels=7).float().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"  Loaded: {checkpoint_path}  (epoch {ckpt.get('epoch','?')}, F1={ckpt.get('dev_f1',0):.4f})")
    else:
        model.load_state_dict(ckpt)
        log.info(f"  Loaded weights: {checkpoint_path}")
    return model


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    root       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev_path   = args.dev_path   or os.path.join(root, "data/processed/dev.csv")
    test_path  = args.test_path  or os.path.join(root, "data/processed/test.csv")
    dev_out    = args.dev_out    or os.path.join(root, "dev.txt")
    test_out   = args.test_out   or os.path.join(root, "test.txt")

    # Threshold
    thresh_file = args.threshold_file or os.path.join(
        root, "RoBERTa/checkpoints/best_threshold.txt"
    )
    if os.path.exists(thresh_file):
        with open(thresh_file) as fh:
            threshold = float(fh.read().strip())
        log.info(f"Threshold from file: {threshold:.4f}")
    else:
        threshold = args.threshold
        log.info(f"Using fallback threshold: {threshold:.4f}")

    # Checkpoints
    checkpoints = args.checkpoints
    if not checkpoints:
        default = os.path.join(root, "RoBERTa/checkpoints/best_model.pt")
        if not os.path.exists(default):
            raise FileNotFoundError(f"No checkpoint at {default}. Run train.py first.")
        checkpoints = [default]
    log.info(f"Ensemble: {len(checkpoints)} checkpoint(s)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Dev
    log.info("Dev inference …")
    dev_df = pd.read_csv(dev_path)
    dev_ds = PCLDataset(dev_df, tokenizer, max_length=args.max_length, is_test=False)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    dev_probs_list = []
    for ckpt in checkpoints:
        m = load_model(ckpt, args.model_name, device)
        dev_probs_list.append(get_probs(m, dev_loader, device))
        del m; torch.cuda.empty_cache()

    dev_probs = np.mean(dev_probs_list, axis=0)
    dev_preds = (dev_probs >= threshold).astype(int)

    true_labels = dev_ds.binary_labels.numpy()
    f1   = f1_score(true_labels, dev_preds, pos_label=1, zero_division=0)
    prec = precision_score(true_labels, dev_preds, pos_label=1, zero_division=0)
    rec  = recall_score(true_labels, dev_preds, pos_label=1, zero_division=0)
    log.info(f"Dev — F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")

    np.savetxt(dev_out, dev_preds, fmt="%d")
    log.info(f"Saved → {dev_out}  ({int(dev_preds.sum())} PCL)")

    # Test
    log.info("Test inference …")
    test_df = pd.read_csv(test_path)
    test_ds = PCLDataset(test_df, tokenizer, max_length=args.max_length, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_probs_list = []
    for ckpt in checkpoints:
        m = load_model(ckpt, args.model_name, device)
        test_probs_list.append(get_probs(m, test_loader, device))
        del m; torch.cuda.empty_cache()

    test_probs = np.mean(test_probs_list, axis=0)
    test_preds = (test_probs >= threshold).astype(int)

    np.savetxt(test_out, test_preds, fmt="%d")
    log.info(f"Saved → {test_out}  ({int(test_preds.sum())} PCL, {len(test_preds)} total)")

    if len(test_preds) != 3832:
        log.warning(f"test.txt has {len(test_preds)} lines; expected 3832.")

    log.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="RoBERTa-base PCL inference")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--checkpoints",    nargs="+",  default=None)
    p.add_argument("--model_name",     default="roberta-base")
    p.add_argument("--max_length",     type=int,   default=256)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--threshold",      type=float, default=0.5)
    p.add_argument("--threshold_file", default=None)
    p.add_argument("--dev_path",       default=None)
    p.add_argument("--test_path",      default=None)
    p.add_argument("--dev_out",        default=None)
    p.add_argument("--test_out",       default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)

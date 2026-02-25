"""
dataset.py — PCL Dataset with keyword injection and multi-label parsing.

Each sample is formatted as:
    "[{keyword}] {text}"

The label column is a string like "[1, 0, 0, 1, 0, 0, 0]".
  - binary_label  = 1 if any element > 0, else 0
  - multi_label   = the 7-element vector (for the auxiliary head)
"""

import ast
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class PCLDataset(Dataset):
    """
    Dataset for binary PCL detection with optional 7-label auxiliary targets.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ["text", "keyword"].
        For train/dev also needs "label" (7-element list string).
    tokenizer : PreTrainedTokenizerFast
    max_length : int
    is_test : bool
        If True, no label column is expected and labels are returned as -1.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 128,
        is_test: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Build prefix-injected texts: "[keyword] original text"
        # Explicit str() cast ensures plain Python str, not numpy str_ —
        # transformers 5.x isinstance check rejects numpy scalar strings.
        self.texts = [
            f"[{str(kw)}] {str(txt)}"
            for kw, txt in zip(df["keyword"].fillna(""), df["text"].fillna(""))
        ]

        if not is_test:
            parsed = df["label"].apply(self._parse_label)
            self.multi_labels = torch.tensor(
                [x for x in parsed], dtype=torch.float32
            )  # (N, 7)
            self.binary_labels = (self.multi_labels.sum(dim=1) > 0).long()  # (N,)
        else:
            n = len(df)
            self.multi_labels = torch.full((n, 7), -1, dtype=torch.float32)
            self.binary_labels = torch.full((n,), -1, dtype=torch.long)

    @staticmethod
    def _parse_label(label_str: str):
        """Parse '[1, 0, 0, 1, 0, 0, 0]' → [1, 0, 0, 1, 0, 0, 0]."""
        try:
            return ast.literal_eval(str(label_str))
        except Exception:
            return [0] * 7

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # guarantee plain str
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "binary_label": self.binary_labels[idx],
            "multi_label": self.multi_labels[idx],
        }

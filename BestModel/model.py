"""
model.py — Dual-head PCL model built on RoBERTa
Architecture
------------
  RoBERTa-base encoder
        │
   [CLS] token → Dropout
        ├─→ Linear(hidden, 1)  ← binary PCL head   (scalar logit)
        └─→ Linear(hidden, 7)  ← auxiliary 7-label head (7 logits)

The binary head is the primary output; the auxiliary head provides extra
supervision from the fine-grained PCL-category labels during training.
At inference time only the binary head is used.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter.  gamma=0 → standard BCE.  gamma=2 is canonical.
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N,) raw (un-sigmoid'd) model output
        targets : (N,) float binary labels ∈ {0.0, 1.0}
        """
        # Standard BCE (per-element, no reduction)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # p_t = probability of the true class
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PCLModel(nn.Module):
    """
    RoBERTa-base with:
      • binary head  (primary task)
      • 7-label auxiliary multi-task head

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier
    num_aux_labels : int
        Number of fine-grained PCL categories (7 for Don't Patronize Me!).
    dropout : float
        Dropout probability applied to [CLS] before both heads.
    """

    def __init__(
        self,
        model_name: str = "FacebookAI/roberta-base",
        num_aux_labels: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden_size, 1)
        self.aux_head = nn.Linear(hidden_size, num_aux_labels)

        # Initialise output heads with small weights for training stability
        nn.init.normal_(self.binary_head.weight, std=0.02)
        nn.init.zeros_(self.binary_head.bias)
        nn.init.normal_(self.aux_head.weight, std=0.02)
        nn.init.zeros_(self.aux_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Returns
        -------
        binary_logits : (N,)   — raw logit for PCL positive class
        aux_logits    : (N, 7) — raw logits for 7-category auxiliary task
        """
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] representation (position 0)
        cls_repr = encoder_out.last_hidden_state[:, 0, :]
        cls_repr = self.dropout(cls_repr)

        binary_logits = self.binary_head(cls_repr).squeeze(-1)  # (N,)
        aux_logits = self.aux_head(cls_repr)                     # (N, 7)

        return binary_logits, aux_logits

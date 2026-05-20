
"""
Whole-Data Decorrelation Within-Class Correlation (WDDWCC).

A correlation-based supervised DR method that provides a direct
counterpart to WDIWCD for ablation comparison (Section V-B, Table 7).

Core idea
---------
Instead of independence/dependence criteria (as in WDIWCD),
this method uses dot-product correlation:
  - Minimise the correlation between the projected feature and its
    null-space projection across the whole dataset (decorrelation).
  - Maximise the correlation within each class (within-class correlation).

Because correlation is a weaker criterion than statistical independence,
WDDWCC typically yields lower contrast and interpretability scores than
WDIWCD, as reported in Tables 6–7 of the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WDDWCC(nn.Module):
    """Linear WDDWCC extractor."""

    def __init__(self, input_dim: int, n_components: int = 16):
        super().__init__()
        # Learnable projection vector
        w_init = torch.randn(input_dim, n_components) * 0.01
        self.w = nn.Parameter(w_init)
        # Fixed random null-space combiner
        a_init = torch.randn(1, input_dim)
        a_init = F.normalize(a_init, dim=1)
        self.register_buffer('a_k', a_init)

    def _null_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Orthogonal complement projection."""
        w_n   = F.normalize(self.w, dim=0)
        W_n   = torch.eye(x.shape[1], device=x.device) - w_n @ w_n.t()
        return (self.a_k @ W_n.t() @ x.t()).t()

    def compute_loss(
        self, x: torch.Tensor, labels: torch.Tensor,
        per_class_weight: float = 0.5,
    ) -> torch.Tensor:
        """WDDWCC objective: global decorrelation + within-class correlation."""
        x1 = x @ self.w
        x2 = self._null_projection(x)

        # Whole-data decorrelation term (minimise dot product)
        dot_all   = (x1 * x2).sum() / max(x1.shape[0], 1)
        loss_whole = dot_all ** 2

        # Within-class correlation term (maximise dot product per class)
        loss_class = torch.tensor(0.0, device=x.device)
        for cls in labels.unique():
            mask = labels == cls
            if mask.sum() < 2:
                continue
            dot_cls    = (x1[mask] * x2[mask]).sum() / mask.sum().float()
            loss_class -= dot_cls ** 2 / len(labels.unique())

        return (1.0 - per_class_weight) * loss_whole + per_class_weight * loss_class

    def transform(self, x: torch.Tensor) -> np.ndarray:
        return (x @ self.w).detach().cpu().numpy()


"""
Whole-Data Independence Within-Class Dependence (WDIWCD).

This module implements the proposed supervised dimensionality reduction
method described in Section III-B of the paper.

Core idea
---------
A single linear projection vector `w` is learned such that:
  - The projected whole-dataset distribution is as *independent* as possible
    (encourages between-class contrast and diversity).
  - The projected within-class distributions are as *dependent* as possible
    (encourages within-class cluster cohesion).

Both objectives are measured using fuzzy-histogram approximations of marginal
and joint probability density functions, making the criterion differentiable
and amenable to gradient-based optimisation.

The probability product rule (PPR) criterion for independence is:
    loss_independence = E[ (p(x1, x2) - p(x1) * p(x2))^2 ]
which is maximised for the whole dataset and minimised within each class.
"""
import math
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F


def xavier_uniform_init(shape):
    """Xavier uniform initialisation for a weight tensor of given shape."""
    bound = math.sqrt(6.0) / math.sqrt(2.0 * shape[1])
    return np.random.uniform(-bound, bound, size=shape)


class WDIWCD(nn.Module):
    """Linear WDIWCD extractor (one projection vector per component)."""

    def __init__(
        self,
        input_dim: int,
        n_components: int = 16,
        hist_bins: int = 32,
        n_classes: int = 10,
        sigma_factor: float = 10.0,
        membership: str = 'gauss',
    ):
        super().__init__()
        self.n_components = n_components
        self.hist_bins    = hist_bins
        self.n_classes    = n_classes
        self.sigma        = sigma_factor / hist_bins

        # Learnable projection vector (space direction)
        w_init = xavier_uniform_init((input_dim, n_components))
        self.w = nn.Parameter(t.tensor(w_init, dtype=t.float32))

        # Random null-space combiner (fixed; used to span orthogonal direction)
        a_init = np.random.randn(1, input_dim)
        a_init /= np.linalg.norm(a_init)
        self.register_buffer('a_k', t.tensor(a_init, dtype=t.float32))

        # Fuzzy membership functions for histogram approximation
        self._gauss = lambda d: t.exp(-(d ** 2) / (self.sigma ** 2))

    # ── helpers ───────────────────────────────────────────────────────────
    def _null_projection(self, data: t.Tensor) -> t.Tensor:
        """Project data onto the null-space of w (orthogonal complement)."""
        w_norm = self.w / (self.w.norm(dim=0, keepdim=True) + 1e-8)
        W_null = t.eye(self.w.shape[0], device=self.w.device) - w_norm.mm(w_norm.t())
        return (self.a_k.mm(W_null.t()).mm(data.t())).t()

    def _hist1d(self, x: t.Tensor) -> t.Tensor:
        """Compute a soft 1-D histogram of x using Gaussian fuzzy membership."""
        x_min, x_max = x.min().item(), x.max().item()
        centres = t.linspace(x_min, x_max, self.hist_bins,
                             device=x.device).unsqueeze(0)  # (1, M)
        diff    = x - centres                               # (N, M)
        return self._gauss(diff)                            # (N, M)

    @staticmethod
    def _marginal_joint(h1: t.Tensor, h2: t.Tensor):
        """Return (marginal_product, joint) histograms from two soft histograms."""
        M = h1.shape[1]
        # Marginal product: outer product of column sums
        s1  = h1.sum(0, keepdim=True)   # (1, M)
        s2  = h2.sum(0, keepdim=True)   # (1, M)
        mrg = (s1.t() @ s2).view(-1)    # (M*M,)
        # Joint: sum of element-wise products with Kronecker structure
        eye_M  = t.eye(M, device=h1.device)
        ones_M = t.ones(1, M, device=h1.device)
        jnt = (h1 @ t.kron(eye_M,  ones_M.t()) *
               h2 @ t.kron(ones_M.t(), eye_M)).sum(0)
        return mrg, jnt

    # ── forward pass / loss computation ───────────────────────────────────
    def compute_loss(
        self,
        x: t.Tensor,
        labels: t.Tensor,
        perclass_weight: float = 0.7,
    ) -> t.Tensor:
        """Compute the WDIWCD objective.

        The total loss maximises whole-data independence (contrast)
        while simultaneously maximising within-class dependence (cohesion).

        Parameters
        ----------
        x               : Input feature matrix (batch, input_dim).
        labels          : Integer class labels (batch,).
        perclass_weight : Relative weight of the within-class term [0, 1].
        """
        # Project data onto learned direction and its null-space
        x1 = x.mm(self.w)          # (N, n_components) — principal projection
        x2 = self._null_projection(x)  # (N, n_components) — null-space

        # Whole-data independence loss (minimise dependence → maximise independence)
        h1_all = self._hist1d(x1[:, 0:1])
        h2_all = self._hist1d(x2[:, 0:1])
        mrg_all, _ = self._marginal_joint(h1_all, h2_all)
        loss_whole = -t.mean(mrg_all ** 2)   # negative → maximise independence

        # Within-class dependence loss (maximise dependence per class)
        loss_class = t.tensor(0.0, device=x.device)
        classes    = labels.unique()
        for cls in classes:
            mask    = labels == cls
            x1_cls  = x1[mask, 0:1]
            x2_cls  = x2[mask, 0:1]
            if x1_cls.shape[0] < 2:
                continue
            h1_cls = self._hist1d(x1_cls)
            h2_cls = self._hist1d(x2_cls)
            mrg_cls, _ = self._marginal_joint(h1_cls, h2_cls)
            loss_class += t.mean(mrg_cls ** 2) / len(classes)  # maximise

        return (1.0 - perclass_weight) * loss_whole + perclass_weight * (-loss_class)

    def transform(self, x: t.Tensor) -> t.Tensor:
        """Project input onto learned components (inference only)."""
        return x.mm(self.w).detach().cpu().numpy()

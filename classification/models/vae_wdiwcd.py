
"""
VAE with layer-shared WDIWCD (VAE-NWDIWCD).

Architecture (Section III-C, Figure layer-sharing schematic)
------------------------------------------------------------
The first linear layer of the VAE encoder is *shared* with the WDIWCD
module.  This means both objectives backpropagate through the same
weight matrix, enabling the encoder to simultaneously learn:
  (a) a good latent representation for reconstruction (VAE objective), and
  (b) between-sample contrast and within-class cohesion (WDIWCD objective).

Combined loss (Equation 28 in paper)
-------------------------------------
L_total = (1-a)(1-b) * L_recon
        + (1-a)   b  * L_KLD
        +    a    c  * L_wdiwcd_whole_data
        +    a (1-c) * L_wdiwcd_per_class

Optimal hyperparameters from paper (Table, hyperparameter search):
  a = 0.80  (weight of the WDIWCD terms)
  b = 0.48  (relative weight of KLD within the VAE terms)
  c = 0.87  (relative weight of whole-data independence term)
"""
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .wdiwcd import WDIWCD


class VAEWDIWCDEncoder(nn.Module):
    """Encoder whose first layer is shared with the WDIWCD module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_classes: int,
        hist_bins: int = 32,
        n_heads: int = 1,
    ):
        super().__init__()
        self.n_classes  = n_classes
        self.n_heads    = n_heads

        # Shared first layer (used by both VAE encoder and WDIWCD)
        self.shared_layer    = nn.Linear(input_dim, hidden_dim)
        # WDIWCD discriminative block (applied after shared layer)
        self.discrim_layer   = nn.Linear(hidden_dim, hidden_dim)
        # Projection vector learned by the WDIWCD criterion
        self.wdiwcd = WDIWCD(
            input_dim   = hidden_dim,
            n_components= max(1, hidden_dim // n_heads),
            hist_bins   = hist_bins,
            n_classes   = n_classes,
        )
        # VAE latent-space heads
        self.fc_half = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu   = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logv = nn.Linear(hidden_dim // 2, latent_dim)
        # Softmax classifier head (categorical cross-entropy auxiliary loss)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: t.Tensor, labels: t.Tensor):
        # Shared encoding
        h      = F.relu(self.shared_layer(x))
        h_disc = F.relu(self.discrim_layer(h))

        # VAE latent distribution parameters
        h2     = F.relu(self.fc_half(h))
        mu     = self.fc_mu(h2)
        log_var = self.fc_logv(h2)
        sigma  = torch.exp(0.5 * log_var)

        # Reparameterise
        if self.training:
            z = mu + torch.randn_like(sigma) * sigma
        else:
            z = mu

        # WDIWCD independence criterion loss
        wdiwcd_loss = self.wdiwcd.compute_loss(h_disc, labels)

        # Auxiliary softmax classification logits
        logits = self.classifier(h)

        return z, mu, log_var, wdiwcd_loss, logits, h_disc


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim,      hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, z):
        return self.fc2(F.relu(self.fc1(z)))


class VAE_WDIWCD(nn.Module):
    """Full VAE-NWDIWCD model with layer sharing."""

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 500,
        latent_dim: int = 16,
        n_classes:  int = 10,
        hist_bins:  int = 32,
        n_heads:    int = 1,
    ):
        super().__init__()
        self.encoder = VAEWDIWCDEncoder(
            input_dim, hidden_dim, latent_dim, n_classes, hist_bins, n_heads)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x, labels):
        z, mu, log_var, wdiwcd_loss, logits, h_disc = self.encoder(x, labels)
        recon = self.decoder(z)
        return recon, mu, log_var, wdiwcd_loss, logits, z

    @staticmethod
    def combined_loss(
        recon, x, mu, log_var, wdiwcd_loss,
        a: float = 0.80,
        b: float = 0.48,
    ):
        """Combined loss L_total (Equation 28).

        Parameters
        ----------
        a : weight of WDIWCD terms (optimal = 0.80)
        b : weight of KLD within VAE terms (optimal = 0.48)
        Note: the within-class / whole-data split (c=0.87) is
              handled inside WDIWCD.compute_loss via perclass_weight.
        """
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        vae_term   = (1.0 - b) * recon_loss + b * kl_loss
        return (1.0 - a) * vae_term + a * wdiwcd_loss

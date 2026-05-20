
"""
Baseline Variational Autoencoder (VAE).

Architecture (as described in Section IV of the paper):
  Encoder: Input → 512 (ReLU) → 256 (ReLU) → [mu, log_sigma] (latent_dim)
  Decoder: latent_dim → 256 (ReLU) → 512 (ReLU) → Output (sigmoid)

Loss = MSE reconstruction + beta * KL divergence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1     = nn.Linear(input_dim,  hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu   = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logv = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logv(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim,      hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)          # raw logits; apply sigmoid outside if needed


class VAE(nn.Module):
    """Full VAE combining encoder and decoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, latent_dim: int = 16):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)

    def reparameterise(self, mu, log_var):
        """Reparameterisation trick: z = mu + eps * sigma, eps ~ N(0, I)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterise(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var, z


def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0):
    """Combined VAE loss: MSE reconstruction + beta-weighted KL divergence."""
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# gated_model_def_Unbiased.py
# -*- coding: utf-8 -*-

# Implements a Gated VAE architecture for binary and continuous mixed data

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Definition of the gated model's architecture 
class GatedMixedVAE(nn.Module):
    # Initialise the VAE model architecture - encoder, decoder and perceptrons
    def __init__(self, input_dim=67, latent_dim=11,
                 enc_hidden=(256, 128, 64),
                 dec_hidden=(64, 128, 256),
                 binary_idx=None,
                 cont_idx=None,
                 clamp_latent_logvar: bool = False,
                 latent_logvar_min: float = -2.0,
                 latent_logvar_max: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.binary_idx = torch.as_tensor(binary_idx, dtype=torch.long)
        self.cont_idx   = torch.as_tensor(cont_idx,   dtype=torch.long)

        self.clamp_latent_logvar = bool(clamp_latent_logvar)
        self.latent_logvar_min = float(latent_logvar_min)
        self.latent_logvar_max = float(latent_logvar_max)

        # Encoder definition - layers: 256, 128, 64 -> 11
        enc_sizes = [input_dim] + list(enc_hidden)
        self.enc_layers = nn.ModuleList([
            nn.Linear(enc_sizes[i], enc_sizes[i + 1])
            for i in range(len(enc_sizes) - 1)
        ])
        enc_last = enc_sizes[-1]
        self.mu     = nn.Linear(enc_last, latent_dim)
        self.logvar = nn.Linear(enc_last, latent_dim)

        # Decoder - Layers: 11 -> 64, 128, 256
        dec_sizes = [latent_dim] + list(dec_hidden)
        self.dec_layers = nn.ModuleList([
            nn.Linear(dec_sizes[i], dec_sizes[i + 1])
            for i in range(len(dec_sizes) - 1)
        ])
        dec_last = dec_sizes[-1] if len(dec_sizes) > 1 else latent_dim

        # Binary, gate and continuous heads
        self.bin_head   = nn.Linear(dec_last, len(self.binary_idx))
        self.gate_head  = nn.Linear(dec_last, len(self.cont_idx))
        self.cont_head  = nn.Linear(dec_last, 2 * len(self.cont_idx))

        self.apply(self._xavier_init)

    # Initialises Xavier for training, weighting across layers
    @staticmethod
    def _xavier_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Defines the encoded inputs into mean and logvar
    def encode(self, x):
        h = x
        for layer in self.enc_layers:
            h = F.relu(layer(h))
        mu = self.mu(h)
        logvar = self.logvar(h)

        if self.clamp_latent_logvar:
            logvar = torch.clamp(logvar, min=self.latent_logvar_min, max=self.latent_logvar_max)

        # Outputs for latent distribution - mu and logvar
        return mu, logvar

    # Reparamterisation trick for sampling from the latent distribution
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _dec_body(self, z):
        h = z
        for layer in self.dec_layers:
            h = F.relu(layer(h))
        return h

    # Decodes the samples into the binray logits, continuous mean & variance, and the gating logits for coefficients
    def decode(self, z):
        h = self._dec_body(z)
        bin_logits  = self.bin_head(h)
        gate_logits = self.gate_head(h)

        cont_params = self.cont_head(h).view(-1, len(self.cont_idx), 2)
        cont_mu   = cont_params[..., 0]
        raw_var   = cont_params[..., 1]

        sigma_min = 0.1
        cont_var = sigma_min**2 + F.softplus(raw_var)
        cont_logvar = torch.log(cont_var)
        return bin_logits, cont_mu, cont_logvar, gate_logits

    # Defines the full pass of the encoder to the decoder
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        bin_logits, cont_mu, cont_logvar, gate_logits = self.decode(z)
        return bin_logits, cont_mu, cont_logvar, gate_logits, mu, logvar

# Computes the BCE, Gate prediction loss, NLL and KL Divergence
def gated_mixed_vae_loss(x_target,
                         bin_logits,
                         cont_mu, cont_logvar,
                         gate_logits,
                         mu, logvar,
                         binary_idx, cont_idx,
                         beta=0.1):
    x_bin  = x_target[:, binary_idx]
    x_cont = x_target[:, cont_idx]

    bce_bin = F.binary_cross_entropy_with_logits(
        bin_logits, x_bin, reduction='none'
    ).sum(dim=1)

    gate_target = (torch.abs(x_cont) > 0).float()
    gate_bce = F.binary_cross_entropy_with_logits(
        gate_logits, gate_target, reduction='none'
    ).sum(dim=1)

    cont_var = torch.exp(cont_logvar)
    nll_per_dim = 0.5 * (
        math.log(2 * math.pi) + cont_logvar +
        ((x_cont - cont_mu).pow(2) / cont_var)
    )
    nll = (nll_per_dim * gate_target).sum(dim=1)

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # Combine all terms with KL weighting - using Beta value
    loss = bce_bin + gate_bce + nll + beta * kl
    return loss.mean(), bce_bin.mean(), gate_bce.mean(), nll.mean(), kl.mean()

# Converts each row into a viewable string of terms
def pde_row_to_string(row: np.ndarray,
                      labels: list[str],
                      binary_idx: np.ndarray,
                      cont_idx: np.ndarray,
                      zero_tol: float = 1e-6,
                      coef_fmt: str = "{:+.3g}") -> str:
    terms = []

    # Continuous terms first
    for i in cont_idx:
        val = float(row[i])
        if abs(val) > zero_tol:
            coef = coef_fmt.format(val)
            if not terms and coef.startswith('+'):
                coef = coef[1:]
            terms.append(f"{coef}{labels[i]}")

    # Binary terms
    for i in binary_idx:
        if row[i] >= 1 - zero_tol:
            prefix = "" if not terms else "+ "
            terms.append(f"{prefix}{labels[i]}")

    if not terms:
        return "0"

    expr = " ".join(terms).replace("+  ", "+ ")
    return expr
import os
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from gated_model_def_Unbiased import GatedMixedVAE, gated_mixed_vae_loss


# Definiton of the latent, encoder and decoder layers
Latent_Dim = 11
enc_hidden = (256, 128, 64)
dec_hidden = (64, 128, 256)

# Epochs define the number of training cycles, the batch size and the learning rate for the optimiser
epochs = 150
batch_size = 64
lr = 1e-3

# The annealing parameters for the Beta value of 0.1, anneals up across 80 training epochs
beta_final = 0.1
anneal_epochs = 80
anneal_mode = "cosine"

# Latent logvar clamp range
LATENT_LOGVAR_MIN = -2.0
LATENT_LOGVAR_MAX = 2.0

# Output files for the saved data
CKPT_PATH   = "mixed_vae_gated_checkpoint_unbiased.pth"
XMIX_PATH   = "X_mixed_unbiased.npy"
COUNTS_PATH = "counts_active_unbiased.npy"


# 67 terms library used - quick sanity check to ensure the term library matches the count 
n_terms = 67

term_labels = [
    "Ut", "Utt", "U",
    "Ux", "Uy", "Uz",
    "Uxx", "Uyy", "Uzz", "Uxy", "Uxz", "Uyz",

    "UU", "UUx", "UUy", "UUz",
    "UUxx", "UUyy", "UUzz", "UUxy", "UUxz", "UUyz",

    "UxUx", "UxUy", "UxUz", "UxUxx", "UxUyy", "UxUzz", "UxUxy", "UxUxz", "UxUyz",
    "UyUy", "UyUz", "UyUxx", "UyUyy", "UyUzz", "UyUxy", "UyUxz", "UyUyz",
    "UzUz", "UzUxx", "UzUyy", "UzUzz", "UzUxy", "UzUxz", "UzUyz",

    "UxxUxx", "UxxUyy", "UxxUzz", "UxxUxy", "UxxUxz", "UxxUyz",
    "UyyUyy", "UyyUzz", "UyyUxy", "UyyUxz", "UyyUyz",
    "UzzUzz", "UzzUxy", "UzzUxz", "UzzUyz",

    "UxyUxy", "UxyUxz", "UxyUyz",
    "UxzUxz", "UxzUyz",
    "UyzUyz",
]
assert len(term_labels) == n_terms

cont_idx   = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=int)
binary_idx = np.setdiff1d(np.arange(n_terms), cont_idx)


# Definition of the Beta annealing function - helps to prevent the KL Divergence term from overpowering the loss so reconstruction learned by VAE
def kl_beta_schedule(epoch: int, beta_final: float, anneal_epochs: int, mode: str = "cosine") -> float:
    if anneal_epochs <= 0:
        return beta_final
    t = min(max(epoch, 1), anneal_epochs) / anneal_epochs
    if mode == "linear":
        return beta_final * t
    elif mode == "cosine":
        return beta_final * (0.5 - 0.5 * np.cos(np.pi * t))
    else:
        raise ValueError(f"Unknown anneal mode: {mode}")


# loads the data for trianing the model and saves the checkpoint - combination of generated data and Gated VAE
def main():
    if not os.path.exists(XMIX_PATH):
        raise FileNotFoundError(
            f"Could not find {XMIX_PATH}. Run generate_mixed_unbiased_data.py first."
        )

    if not os.path.exists(COUNTS_PATH):
        raise FileNotFoundError(
            f"Could not find {COUNTS_PATH}. Run generate_mixed_unbiased_data.py first."
        )

    print(f"[LOAD] Loading {XMIX_PATH} ...")
    X_mixed = np.load(XMIX_PATH)

    print(f"[LOAD] Loading {COUNTS_PATH} ...")
    counts_active = np.load(COUNTS_PATH)

    print(f"[DATA] X_mixed shape = {X_mixed.shape}")
    print(f"[DATA] counts_active shape = {counts_active.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] device = {device}")

    model = GatedMixedVAE(
        input_dim=n_terms,
        latent_dim=Latent_Dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        binary_idx=binary_idx,
        cont_idx=cont_idx,
        clamp_latent_logvar=True,
        latent_logvar_min=LATENT_LOGVAR_MIN,
        latent_logvar_max=LATENT_LOGVAR_MAX
    ).to(device)

    X_mixed_t = torch.tensor(X_mixed, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_mixed_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Switches the model to training mode
    model.train()
    for ep in range(1, epochs + 1):
        beta = kl_beta_schedule(
            ep,
            beta_final=beta_final,
            anneal_epochs=anneal_epochs,
            mode=anneal_mode
        )

        total = total_bce = total_gate_bce = total_nll = total_kl = 0.0

        for (x_batch,) in loader:
            x = x_batch.to(device)

            bin_logits, cont_mu, cont_logvar, gate_logits, mu, logvar = model(x)

            loss, bce_bin, gate_bce, nll, kl = gated_mixed_vae_loss(
                x_target=x,
                bin_logits=bin_logits,
                cont_mu=cont_mu,
                cont_logvar=cont_logvar,
                gate_logits=gate_logits,
                mu=mu,
                logvar=logvar,
                binary_idx=torch.as_tensor(binary_idx, device=device),
                cont_idx=torch.as_tensor(cont_idx, device=device),
                beta=beta
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            total_bce += bce_bin.item()
            total_gate_bce += gate_bce.item()
            total_nll += nll.item()
            total_kl += kl.item()

        # Prints the losses for each epoch step
        print(
            f"Epoch {ep:03d} | beta {beta:.4f} | loss {total/len(loader):.3f} "
            f"| BCE_bin {total_bce/len(loader):.3f} "
            f"| BCE_gate {total_gate_bce/len(loader):.3f} "
            f"| NLL {total_nll/len(loader):.3f} "
            f"| KL {total_kl/len(loader):.3f} "
            f"| mean(logvar) {logvar.mean().item():+.3f}"
        )
        
    # Saves the models learned weights and other data for when needed in reloading
    torch.save({
        "model_state_dict": model.state_dict(),
        "binary_idx": binary_idx,
        "cont_idx": cont_idx,
        "term_labels": term_labels,
        "latent_dim": Latent_Dim,
        "enc_hidden": enc_hidden,
        "dec_hidden": dec_hidden,
        "input_dim": n_terms,
        "clamp_latent_logvar": True,
        "latent_logvar_min": LATENT_LOGVAR_MIN,
        "latent_logvar_max": LATENT_LOGVAR_MAX,
    }, CKPT_PATH)

    print(f"[SAVE] checkpoint -> {CKPT_PATH}")
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()

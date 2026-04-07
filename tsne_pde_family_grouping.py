import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import mplcursors
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

from gated_model_def_Unbiased import GatedMixedVAE, pde_row_to_string


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="mixed_vae_gated_checkpoint_unbiased.pth")
parser.add_argument("--xmix", default="X_mixed_unbiased.npy")
parser.add_argument("--counts", default="counts_active_unbiased.npy")
parser.add_argument("--perplexity", type=float, default=40)
parser.add_argument("--max-old", type=int, default=10000)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Loads the model, parameters and data
chk = torch.load(args.ckpt, map_location=device, weights_only=False)

latent_dim  = chk["latent_dim"]
enc_hidden  = chk["enc_hidden"]
dec_hidden  = chk["dec_hidden"]
binary_idx  = chk["binary_idx"]
cont_idx    = chk["cont_idx"]
term_labels = chk["term_labels"]

input_dim = chk.get("input_dim", len(term_labels))

model = GatedMixedVAE(
    input_dim=input_dim,
    latent_dim=latent_dim,
    enc_hidden=enc_hidden,
    dec_hidden=dec_hidden,
    binary_idx=binary_idx,
    cont_idx=cont_idx
).to(device)

model.load_state_dict(chk["model_state_dict"])
model.eval()

X_mixed = np.load(args.xmix).astype(np.float32)
counts_all = np.load(args.counts)

print(f"[LOAD] dataset shape = {X_mixed.shape}")

N = X_mixed.shape[0]

if N > args.max_old:
    idx = rng.choice(N, size=args.max_old, replace=False)
    X_old = X_mixed[idx]
    counts_old = counts_all[idx]
else:
    X_old = X_mixed
    counts_old = counts_all

print(f"[PLOT] plotting {X_old.shape[0]} PDEs")


# Classification of the PDE families for t-SNE plot

IDX_UT  = 0
IDX_UTT = 1

ORDER1 = [3,4,5]
ORDER2 = [6,7,8,9,10,11]

# Applies the classification labels to equations based on derivative operators present
def classify_family(x):
    # Small cutoff so tiny coefficients aren't accidently cosnidered to be active
    tol = 1e-2

    has_ut  = abs(x[IDX_UT]) > tol
    has_utt = abs(x[IDX_UTT]) > tol

    n_first  = np.sum(np.abs(x[ORDER1]) > tol)
    n_second = np.sum(np.abs(x[ORDER2]) > tol)

    has_first  = n_first > 0
    has_second = n_second > 0

    if (not has_ut) and (not has_utt) and has_second:
        return "elliptic"

    if has_ut and (not has_utt) and has_second:
        return "parabolic"

    if has_utt and (not has_ut) and has_second:
        return "hyperbolic"

    if has_first and not has_second:
        return "first_order"

    return "other"


# Designates the colours that applies to each family

family_colors = {
    "elliptic":"blue",
    "parabolic":"green",
    "hyperbolic":"red",
    "first_order":"purple",
    "other":"gray"
}

families = np.array([classify_family(row) for row in X_old])
colors_family = np.array([family_colors[f] for f in families])


# Encodes the PDE into the latent mean for use in t-SNE plots

@torch.no_grad()
def encode_mu(X):

    X_t = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=args.batch,
        shuffle=False
    )

    mus = []

    for (xb,) in loader:

        mu,_ = model.encode(xb.to(device))
        mus.append(mu.cpu())

    return torch.cat(mus).numpy()


print("[ENCODE] encoding dataset into latent space")

Z = encode_mu(X_old)


# 2D visualisation of high dimension 11D space

print("[TSNE] computing t-SNE")

tsne = TSNE(
    n_components=2,
    perplexity=args.perplexity,
    init="pca",
    random_state=args.seed,
    verbose=1
)

Z_tsne = tsne.fit_transform(Z)


# Extracts the PDE string for each point - displays each terms present in the equation

exprs = [
    pde_row_to_string(X_old[i], term_labels, binary_idx, cont_idx)
    for i in range(X_old.shape[0])
]

# Plots each point by the number of active terms

plt.figure(figsize=(7,6))

scatter_counts = plt.scatter(
    Z_tsne[:,0],
    Z_tsne[:,1],
    s=7,
    c=counts_old,
    cmap="viridis",
    alpha=0.85
)

plt.colorbar(scatter_counts,label="Number of active terms")

plt.title("t-SNE of PDE Latent Space (Coloured by Number of Terms)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# Enables the hovering over points in the visualisation to view the PDE string
cursor1 = mplcursors.cursor(scatter_counts, hover=True)

@cursor1.connect("add")
def on_hover_counts(sel):

    idx = sel.index

    sel.annotation.set_text(
        f"{counts_old[idx]} terms\n{exprs[idx]}"
    )

    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

plt.tight_layout()
plt.show()

# Plots the points based on the PDE family goruping
plt.figure(figsize=(7,6))

scatter_family = plt.scatter(
    Z_tsne[:,0],
    Z_tsne[:,1],
    s=7,
    c=colors_family,
    alpha=0.85
)

legend_elements = [
    Line2D([0],[0],marker='o',color='w',label='Elliptic',markerfacecolor='blue',markersize=8),
    Line2D([0],[0],marker='o',color='w',label='Parabolic',markerfacecolor='green',markersize=8),
    Line2D([0],[0],marker='o',color='w',label='Hyperbolic',markerfacecolor='red',markersize=8),
    Line2D([0],[0],marker='o',color='w',label='First Order',markerfacecolor='purple',markersize=8),
]

plt.legend(handles=legend_elements)

plt.title("t-SNE of PDE Latent Space (Coloured by PDE Family)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

cursor2 = mplcursors.cursor(scatter_family, hover=True)

@cursor2.connect("add")
def on_hover_family(sel):

    idx = sel.index

    sel.annotation.set_text(
        f"{families[idx]}\n{exprs[idx]}"
    )

    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

plt.tight_layout()
plt.show()
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gated_model_def_Unbiased import GatedMixedVAE, pde_row_to_string

# Defines the arguments so the script can be reused with different datasets
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="mixed_vae_gated_checkpoint_unbiased.pth")
    p.add_argument("--xmix", type=str, default="X_mixed_unbiased.npy")

    p.add_argument("--n-new", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: auto.")

    # Radius estimation
    p.add_argument("--norm-subset", type=int, default=100_000,
                   help="How many training points to estimate p99 radius. 0 = all.")
    p.add_argument("--radius-percentile", type=float, default=99.0)

    # Outputs
    p.add_argument("--missing-gaussian", type=str, default="missing_gaussian.txt")
    p.add_argument("--missing-ball", type=str, default="missing_ball.txt")
    p.add_argument("--novel-gaussian", type=str, default="novel_gaussian.txt")
    p.add_argument("--novel-ball", type=str, default="novel_ball.txt")
    p.add_argument("--print-missing", type=int, default=0)
    p.add_argument("--print-novel", type=int, default=20)

    # Theory diagnostics
    p.add_argument("--print-theory-examples", type=int, default=20,
                   help="How many theoretical valid support examples to print.")
    p.add_argument("--print-theory-missing-from-train", type=int, default=20,
                   help="How many theoretically valid supports absent from train to print.")
    p.add_argument("--print-theory-missing-from-observed", type=int, default=20,
                   help="How many theoretically valid supports absent from (train + generated valid) to print.")

    return p.parse_args()

# Packs each binary row into bytes so rows can be compared efficiently
def pack_rows_to_void(X_bin: np.ndarray) -> np.ndarray:
    packed = np.packbits(X_bin.astype(np.uint8), axis=1)
    void_dt = np.dtype((np.void, packed.shape[1]))
    return packed.view(void_dt).ravel()

# Converts the packed row representations back into a binary matrix
def unpack_void_rows(void_rows: np.ndarray, n_cols: int) -> np.ndarray:
    if len(void_rows) == 0:
        return np.zeros((0, n_cols), dtype=np.uint8)

    byte_len = void_rows.dtype.itemsize
    packed = np.frombuffer(void_rows.tobytes(), dtype=np.uint8).reshape(-1, byte_len)
    unpacked = np.unpackbits(packed, axis=1)
    return unpacked[:, :n_cols].astype(np.uint8)

# This defines the sampling for the uniform hyperspherical ball 
def sample_uniform_ball(n: int, dim: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    u = u / norms

    U = rng.random(size=(n, 1)).astype(np.float32)
    rho = float(radius) * np.power(U, 1.0 / float(dim))
    return u * rho

# Once again run the encoder over the input rows so as to obtain mu for estimation of the latent radius of the data
@torch.no_grad()
def encode_mu(model: GatedMixedVAE, x_np: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    x_t = torch.tensor(x_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t), batch_size=batch_size, shuffle=False, drop_last=False)

    mus = []
    for (xb,) in loader:
        mu, _ = model.encode(xb.to(device))
        mus.append(mu.detach().cpu())
    return torch.cat(mus, dim=0).numpy()

# This code decodes the latent vectors into the binary active terms across the equations
@torch.no_grad()
def decode_to_active_terms_stochastic(
    model: GatedMixedVAE,
    z_np: np.ndarray,
    binary_idx: np.ndarray,
    cont_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    
    input_dim = model.input_dim
    bin_idx_t = torch.as_tensor(binary_idx, dtype=torch.long, device=device)
    cont_idx_t = torch.as_tensor(cont_idx, dtype=torch.long, device=device)

    Z = torch.from_numpy(z_np)
    N = Z.shape[0]
    out = np.zeros((N, input_dim), dtype=np.uint8)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        zb = Z[start:end].to(device)

        bin_logits, cont_mu, cont_logvar, gate_logits = model.decode(zb)
        bin_probs = torch.sigmoid(bin_logits)
        gate_probs = torch.sigmoid(gate_logits)

        bin_active = torch.bernoulli(bin_probs).to(torch.uint8)
        gate_active = torch.bernoulli(gate_probs).to(torch.uint8)

        xb = torch.zeros((end - start, input_dim), dtype=torch.uint8, device=device)
        xb[:, bin_idx_t] = bin_active
        xb[:, cont_idx_t] = gate_active

        out[start:end] = xb.cpu().numpy()

    return out

# Definitions of the plausibility and physics filters for determination of the 
# equations that are valid/invalid
def make_rule_helpers(term_labels, nonlinear_idx):
    order1 = [3, 4, 5] # first-order operators
    order2 = [6, 7, 8, 9, 10, 11] # second-order operators

    SECOND_DERIV_TOKENS = ("Uxx", "Uyy", "Uzz", "Uxy", "Uxz", "Uyz")
    FIRST_DERIV_TOKENS = ("Ux", "Uy", "Uz")

    token_to_linear_index = {
        "Ux": 3, "Uy": 4, "Uz": 5,
        "Uxx": 6, "Uyy": 7, "Uzz": 8,
        "Uxy": 9, "Uxz": 10, "Uyz": 11,
    }
    # Set of nonlinear constraints
    def nonlinear_requires_second_order(j: int) -> bool:
        lbl = term_labels[j]
        return any(tok in lbl for tok in SECOND_DERIV_TOKENS)

    def nonlinear_requires_first_order(j: int) -> bool:
        lbl = term_labels[j]
        return any(tok in lbl for tok in FIRST_DERIV_TOKENS)

    def nonlinear_required_linear_indices(j: int):
        lbl = term_labels[j]
        req = []
        for tok in ("Uxx", "Uyy", "Uzz", "Uxy", "Uxz", "Uyz", "Ux", "Uy", "Uz"):
            if tok in lbl:
                req.append(token_to_linear_index[tok])

        out = []
        for r in req:
            if r not in out:
                out.append(r)
        return out

    # Hand encoded plausibility check on equation structural validity
    def is_physics_plausible_binary(x: np.ndarray) -> bool:
        has_ut = (x[0] == 1)
        has_utt = (x[1] == 1)
        has_u = (x[2] == 1)

        n_first = int(x[order1].sum())
        n_second = int(x[order2].sum())
        has_spatial_1st = (n_first > 0)
        has_spatial_2nd = (n_second > 0)

        if not (has_u or has_ut or has_utt or has_spatial_1st or has_spatial_2nd):
            return False

        if has_ut and has_utt:
            return False

        if n_first > 3 or n_second > 6:
            return False

        # mixed 2nd-order requires diagonals
        if x[9] == 1 and not (x[6] == 1 and x[7] == 1):
            return False
        if x[10] == 1 and not (x[6] == 1 and x[8] == 1):
            return False
        if x[11] == 1 and not (x[7] == 1 and x[8] == 1):
            return False

        # family structure logic
        if (not has_ut) and (not has_utt) and has_spatial_2nd:
            pass
        elif has_ut and (not has_utt) and has_spatial_2nd:
            pass
        elif has_utt and (not has_ut) and has_spatial_2nd:
            pass
        elif has_ut and (not has_spatial_2nd) and has_spatial_1st:
            pass
        elif (not has_ut) and (not has_utt) and (not has_spatial_2nd) and has_spatial_1st:
            pass
        else:
            return False

        nl_count = int(np.count_nonzero(np.array(x)[nonlinear_idx]))
        if nl_count > 1:
            return False

        highest_linear_order = 2 if has_spatial_2nd else (1 if has_spatial_1st else 0)

        for j in nonlinear_idx:
            if x[j] != 1:
                continue

            if nonlinear_requires_second_order(j) and highest_linear_order < 2:
                return False
            if nonlinear_requires_first_order(j) and highest_linear_order < 1:
                return False

            req_lin = nonlinear_required_linear_indices(j)
            for idx_req in req_lin:
                if x[idx_req] != 1:
                    return False

        total_on = int(x.sum())
        if total_on < 2 or total_on > 6:
            return False

        return True
    
    # PDE family identifier
    def classify_family_binary(x: np.ndarray) -> str:
        has_ut = (x[0] == 1)
        has_utt = (x[1] == 1)

        n_first = int(x[order1].sum())
        n_second = int(x[order2].sum())

        has_spatial_1st = (n_first > 0)
        has_spatial_2nd = (n_second > 0)

        if (not has_ut) and (not has_utt) and has_spatial_2nd:
            return "elliptic"
        if has_ut and (not has_utt) and has_spatial_2nd:
            return "parabolic"
        if has_utt and (not has_ut) and has_spatial_2nd:
            return "hyperbolic"
        if has_ut and (not has_spatial_2nd) and has_spatial_1st:
            return "first_order_time"
        if (not has_ut) and (not has_utt) and (not has_spatial_2nd) and has_spatial_1st:
            return "first_order_steady"
        return "other"

    return is_physics_plausible_binary, classify_family_binary

# Counts the number of rows fall into each PDE family
def family_counts(rows: np.ndarray, classify_family_fn) -> dict:
    counts = {}
    for row in rows:
        fam = classify_family_fn(row)
        counts[fam] = counts.get(fam, 0) + 1
    return counts

# Compares the decoded structures vs the training supports to determine
# percentage of the training rows covered/found in the sampled decoded rows
def coverage_and_missing(
    X_phys_train: np.ndarray,
    X_phys_new: np.ndarray,
) -> tuple[float, int, int, np.ndarray, np.ndarray]:

    train_void = pack_rows_to_void(X_phys_train)
    new_void = pack_rows_to_void(X_phys_new)
    new_unique = np.unique(new_void)

    present = np.isin(train_void, new_unique)
    n_present = int(present.sum())
    coverage = 100.0 * n_present / float(X_phys_train.shape[0])

    missing_idx = np.where(~present)[0]
    return coverage, n_present, int(new_unique.shape[0]), missing_idx, present

# Prints the breakdown of the coverage for each PDE family
def family_coverage_report(X_phys_train, present_mask, classify_family_fn, label: str):
    fam_all = {}
    fam_present = {}

    for i, row in enumerate(X_phys_train):
        fam = classify_family_fn(row)
        fam_all[fam] = fam_all.get(fam, 0) + 1
        if present_mask[i]:
            fam_present[fam] = fam_present.get(fam, 0) + 1

    print(f"\n[{label}] Coverage by training-family:")
    for fam in sorted(fam_all.keys()):
        total = fam_all[fam]
        found = fam_present.get(fam, 0)
        pct = 100.0 * found / total if total > 0 else 0.0
        print(f"  {fam:18s}: {found:5d} / {total:5d} = {pct:6.2f}%")

# Identifies the decoded structures that are novel and determines which
# are structurally valid or not
def novel_structure_report(
    X_phys_train: np.ndarray,
    X_phys_new: np.ndarray,
    classify_family_fn,
    is_valid_fn,
    label: str,
):
    train_void = pack_rows_to_void(X_phys_train)
    new_void = pack_rows_to_void(X_phys_new)
    new_unique = np.unique(new_void)

    is_in_train = np.isin(new_unique, train_void)
    novel_void = new_unique[~is_in_train]
    novel_rows = unpack_void_rows(novel_void, X_phys_train.shape[1])

    if novel_rows.shape[0] == 0:
        print(f"\n[{label}] No novel structures.")
        return novel_rows, np.zeros((0,), dtype=bool)

    valid_mask = np.array([is_valid_fn(row) for row in novel_rows], dtype=bool)
    novel_valid = novel_rows[valid_mask]

    print(f"\n[{label}] Novel unique structures:")
    print(f"  total novel unique decoded structures      : {novel_rows.shape[0]}")
    print(f"  structurally valid novel unique structures : {novel_valid.shape[0]}")
    print(f"  structurally invalid novel unique structures: {novel_rows.shape[0] - novel_valid.shape[0]}")

    fam_counts = family_counts(novel_valid, classify_family_fn)
    if novel_valid.shape[0] > 0:
        print(f"\n[{label}] Family breakdown of novel valid structures:")
        for fam in sorted(fam_counts.keys()):
            c = fam_counts[fam]
            pct = 100.0 * c / novel_valid.shape[0]
            print(f"  {fam:18s}: {c:5d} ({pct:6.2f}%)")

    return novel_rows, valid_mask

# Writes any missing training structures that were not found in the generated samples
def write_missing(
    path: str,
    missing_idx: np.ndarray,
    X_phys_train: np.ndarray,
    term_labels,
    binary_idx,
    cont_idx,
    header: str,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        for idx in missing_idx:
            row = X_phys_train[idx]
            if term_labels is not None:
                expr = pde_row_to_string(
                    row.astype(np.float32),
                    term_labels,
                    binary_idx=binary_idx,
                    cont_idx=cont_idx,
                    zero_tol=1e-6
                )
                f.write(f"{idx}\t{expr}\n")
            else:
                f.write(f"{idx}\t{row.tolist()}\n")

# Writes any novel structures, including the structurally valid and invalid ones
def write_novel(
    path: str,
    novel_rows: np.ndarray,
    valid_mask: np.ndarray,
    term_labels,
    binary_idx,
    cont_idx,
    classify_family_fn,
    header: str,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        for i, row in enumerate(novel_rows):
            valid = bool(valid_mask[i])
            fam = classify_family_fn(row) if valid else "invalid"
            expr = pde_row_to_string(
                row.astype(np.float32),
                term_labels,
                binary_idx=binary_idx,
                cont_idx=cont_idx,
                zero_tol=1e-6
            )
            f.write(f"{i}\tvalid={valid}\tfamily={fam}\t{expr}\n")

# Determines the number of valid structures dtermined by the encoded constraints
def enumerate_theoretical_valid_supports(input_dim, is_valid_fn, classify_family_fn):
    valid_rows = []
    nonlinear_idx = list(range(12, input_dim))

    # Enumerate all 2^12 linear blocks
    for lin_mask in range(1 << 12):
        x = np.zeros(input_dim, dtype=np.uint8)

        for k in range(12):
            if (lin_mask >> k) & 1:
                x[k] = 1

        # Case A: no nonlinear term
        if is_valid_fn(x):
            valid_rows.append(x.copy())

        # Case B: exactly one nonlinear term
        for j in nonlinear_idx:
            x[j] = 1
            if is_valid_fn(x):
                valid_rows.append(x.copy())
            x[j] = 0

    if len(valid_rows) == 0:
        valid_rows = np.zeros((0, input_dim), dtype=np.uint8)
    else:
        valid_rows = np.array(valid_rows, dtype=np.uint8)

        # Deduplicate just in case
        valid_void = pack_rows_to_void(valid_rows)
        uniq_void = np.unique(valid_void)
        valid_rows = unpack_void_rows(uniq_void, input_dim)

    fam_counts = family_counts(valid_rows, classify_family_fn)
    return valid_rows, fam_counts

# Prints all the rows of theoretically valid structures and breaks them down into families etc
def print_rows_with_expr(rows, title, max_n, term_labels, binary_idx, cont_idx, classify_family_fn=None):
    print(f"\n--- {title} ---")
    if rows.shape[0] == 0:
        print("(none)")
        return

    n_show = min(max_n, rows.shape[0])
    for i in range(n_show):
        expr = pde_row_to_string(
            rows[i].astype(np.float32),
            term_labels,
            binary_idx=binary_idx,
            cont_idx=cont_idx
        )
        if classify_family_fn is not None:
            fam = classify_family_fn(rows[i])
            print(f"{i}: [{fam}] {expr}")
        else:
            print(f"{i}: {expr}")


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device is None and torch.cuda.is_available()) else "cpu") \
        if args.device is None else torch.device(args.device)

    print(f"[ARGS] n_new={args.n_new} seed={args.seed} device={device}")

    # Loads the checkpoints and rebuilds the model
    chk = torch.load(args.ckpt, map_location=device, weights_only=False)

    latent_dim = int(chk["latent_dim"])
    enc_hidden = chk["enc_hidden"]
    dec_hidden = chk["dec_hidden"]
    binary_idx = np.array(chk["binary_idx"], dtype=int)
    cont_idx = np.array(chk["cont_idx"], dtype=int)
    term_labels = chk.get("term_labels", None)
    input_dim = int(chk.get("input_dim", 67))

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

    print(f"[MODEL] input_dim={input_dim} latent_dim={latent_dim}")

    nonlinear_idx = list(range(12, input_dim))
    is_valid_fn, classify_family_fn = make_rule_helpers(term_labels, nonlinear_idx)

    # Loads the data and converts it to binary structure, removing the mixed aspect of the dataset
    X_mixed = np.load(args.xmix).astype(np.float32)
    if X_mixed.shape[1] != input_dim:
        raise ValueError(f"X_mixed has dim {X_mixed.shape[1]} but model expects {input_dim}")

    X_phys_train = (np.abs(X_mixed) > 0).astype(np.uint8)
    n_train = X_phys_train.shape[0]
    print(f"[DATA] X_mixed={X_mixed.shape} -> X_phys_train={X_phys_train.shape}")

    # Unique training binary structures
    train_void = pack_rows_to_void(X_phys_train)
    train_unique_void = np.unique(train_void)
    n_unique_train_supports = train_unique_void.shape[0]
    print(f"[TRAIN] Unique binary support patterns: {n_unique_train_supports}/{n_train}")

    train_family_counts = family_counts(X_phys_train, classify_family_fn)
    print("\n[TRAIN] Family breakdown of training supports:")
    for fam in sorted(train_family_counts.keys()):
        c = train_family_counts[fam]
        pct = 100.0 * c / n_train
        print(f"  {fam:18s}: {c:5d} ({pct:6.2f}%)")

    # Determines the theoretically valid structures
    theory_valid_rows, theory_family_counts = enumerate_theoretical_valid_supports(
        input_dim=input_dim,
        is_valid_fn=is_valid_fn,
        classify_family_fn=classify_family_fn
    )
   
    theory_void = pack_rows_to_void(theory_valid_rows)
    n_theory = theory_valid_rows.shape[0]
   
    theory_outfile = "all_valid_binary_structures_by_family.txt"

    family_groups = {
        "elliptic": [],
        "parabolic": [],
        "hyperbolic": [],
        "first_order_time": [],
        "first_order_steady": [],
        "other": []
        }

    # Group rows by family
    for row in theory_valid_rows:
        fam = classify_family_fn(row)
        family_groups[fam].append(row)

    with open(theory_outfile, "w", encoding="utf-8") as f:

        f.write("ALL THEORETICALLY VALID BINARY PDE STRUCTURES\n")
        f.write(f"Total structures: {n_theory}\n\n")

        for fam in ["elliptic",
                    "parabolic",
                    "hyperbolic",
                    "first_order_time",
                    "first_order_steady",
                    "other"]:

            rows = family_groups[fam]

            if len(rows) == 0:
                continue

            f.write("=====================================================\n")
            f.write(f"FAMILY: {fam}   (count = {len(rows)})\n")
            f.write("=====================================================\n\n")

            for i, row in enumerate(rows):

                expr = pde_row_to_string(
                    row.astype(np.float32),
                    term_labels,
                    binary_idx=binary_idx,
                    cont_idx=cont_idx
                    )

                f.write(f"{i:5d} : {expr}\n")

            f.write("\n\n")

    print(f"[WRITE] Wrote all theoretical valid structures to: {theory_outfile}")
   

    print(f"\n[THEORY] Total theoretically valid support patterns: {n_theory}")
    print("[THEORY] Family breakdown of theoretically valid supports:")
    for fam in sorted(theory_family_counts.keys()):
        c = theory_family_counts[fam]
        pct = 100.0 * c / n_theory if n_theory > 0 else 0.0
        print(f"  {fam:18s}: {c:5d} ({pct:6.2f}%)")

    # Check how much of theoretical structures are in the training data
    train_in_theory_mask = np.isin(train_void, theory_void)
    n_train_in_theory = int(train_in_theory_mask.sum())
    print(f"\n[CHECK] Training rows valid under theory rules: {n_train_in_theory}/{n_train}")
    print(f"[CHECK] Unique training supports / theory: {n_unique_train_supports}/{n_theory} "
          f"= {100.0 * n_unique_train_supports / n_theory:.2f}%")

    theory_missing_from_train_void = theory_void[~np.isin(theory_void, train_unique_void)]
    theory_missing_from_train_rows = unpack_void_rows(theory_missing_from_train_void, input_dim)
    print(f"[THEORY] Valid supports absent from training: {theory_missing_from_train_rows.shape[0]}")

    # Estimates the latent radius to the 99th percentile from the mu values of the enocder
    if args.norm_subset and args.norm_subset > 0 and args.norm_subset < n_train:
        idx = rng.choice(n_train, size=args.norm_subset, replace=False)
        X_for_norm = X_mixed[idx]
        print(f"[RADIUS] Estimating ||mu(x)|| on subset of {args.norm_subset} / {n_train}...")
    else:
        X_for_norm = X_mixed
        print(f"[RADIUS] Estimating ||mu(x)|| on ALL {n_train} training points...")

    mu = encode_mu(model, X_for_norm, batch_size=min(args.batch, 4096), device=device)
    norms = np.linalg.norm(mu, axis=1)

    pctl = float(args.radius_percentile)
    R = float(np.percentile(norms, pctl))

    print("\n[RADIUS] ||mu(x)|| stats on chosen set:")
    print(f"  min={norms.min():.3f} mean={norms.mean():.3f} median={np.percentile(norms,50):.3f}")
    print(f"  p90={np.percentile(norms,90):.3f} p95={np.percentile(norms,95):.3f} "
          f"p99={np.percentile(norms,99):.3f} max={norms.max():.3f}")
    print(f"[RADIUS] Using ball radius R = p{pctl:.1f} = {R:.6f}\n")

    # Sample z 20 million times from a standard Gaussian
    print(f"[GAUSS] Sampling z ~ N(0, I): n={args.n_new} dim={latent_dim}")
    z_gauss = rng.normal(size=(args.n_new, latent_dim)).astype(np.float32)

    print("[GAUSS] Decoding stochastically...")
    X_gauss = decode_to_active_terms_stochastic(
        model=model,
        z_np=z_gauss,
        binary_idx=binary_idx,
        cont_idx=cont_idx,
        batch_size=args.batch,
        device=device
    )

    cov_g, n_pres_g, n_uniq_g, miss_g, present_g = coverage_and_missing(X_phys_train, X_gauss)
    print(f"[GAUSS] Unique decoded rows: {n_uniq_g}")
    print(f"[GAUSS] Coverage: {cov_g:.2f}% ({n_pres_g}/{n_train})")
    family_coverage_report(X_phys_train, present_g, classify_family_fn, label="GAUSS")

    novel_rows_g, novel_valid_mask_g = novel_structure_report(
        X_phys_train, X_gauss, classify_family_fn, is_valid_fn, label="GAUSS"
    )
    novel_valid_rows_g = novel_rows_g[novel_valid_mask_g]
    novel_valid_void_g = pack_rows_to_void(novel_valid_rows_g) if novel_valid_rows_g.shape[0] > 0 else np.zeros((0,), dtype=np.dtype((np.void, 1)))

    # Sample z 20 million times inside the learned latent hypersphere
    print(f"\n[BALL] Sampling z uniform in ball ||z||<=R: n={args.n_new} dim={latent_dim} R={R:.6f}")
    z_ball = sample_uniform_ball(args.n_new, latent_dim, R, rng)

    print("[BALL] Decoding stochastically...")
    X_ball = decode_to_active_terms_stochastic(
        model=model,
        z_np=z_ball,
        binary_idx=binary_idx,
        cont_idx=cont_idx,
        batch_size=args.batch,
        device=device
    )

    cov_b, n_pres_b, n_uniq_b, miss_b, present_b = coverage_and_missing(X_phys_train, X_ball)
    print(f"[BALL] Unique decoded rows: {n_uniq_b}")
    print(f"[BALL] Coverage: {cov_b:.2f}% ({n_pres_b}/{n_train})")
    family_coverage_report(X_phys_train, present_b, classify_family_fn, label="BALL")

    novel_rows_b, novel_valid_mask_b = novel_structure_report(
        X_phys_train, X_ball, classify_family_fn, is_valid_fn, label="BALL"
    )
    novel_valid_rows_b = novel_rows_b[novel_valid_mask_b]
    novel_valid_void_b = pack_rows_to_void(novel_valid_rows_b) if novel_valid_rows_b.shape[0] > 0 else np.zeros((0,), dtype=np.dtype((np.void, 1)))

    # Compare the observed sapce vs theory
    observed_valid_void_g = np.unique(np.concatenate([train_unique_void, novel_valid_void_g])) \
        if novel_valid_rows_g.shape[0] > 0 else train_unique_void.copy()
    observed_valid_void_b = np.unique(np.concatenate([train_unique_void, novel_valid_void_b])) \
        if novel_valid_rows_b.shape[0] > 0 else train_unique_void.copy()

    n_observed_valid_g = observed_valid_void_g.shape[0]
    n_observed_valid_b = observed_valid_void_b.shape[0]

    print("\n[COMPARE] Observed valid support space vs theory:")
    print(f"  TRAIN only         : {n_unique_train_supports:5d} / {n_theory:5d} = {100.0 * n_unique_train_supports / n_theory:.2f}%")
    print(f"  TRAIN + GAUSS valid: {n_observed_valid_g:5d} / {n_theory:5d} = {100.0 * n_observed_valid_g / n_theory:.2f}%")
    print(f"  TRAIN + BALL  valid: {n_observed_valid_b:5d} / {n_theory:5d} = {100.0 * n_observed_valid_b / n_theory:.2f}%")

    theory_missing_from_observed_g_void = theory_void[~np.isin(theory_void, observed_valid_void_g)]
    theory_missing_from_observed_b_void = theory_void[~np.isin(theory_void, observed_valid_void_b)]

    theory_missing_from_observed_g_rows = unpack_void_rows(theory_missing_from_observed_g_void, input_dim)
    theory_missing_from_observed_b_rows = unpack_void_rows(theory_missing_from_observed_b_void, input_dim)

    print(f"[COMPARE] Theory supports unseen after TRAIN + GAUSS valid: {theory_missing_from_observed_g_rows.shape[0]}")
    print(f"[COMPARE] Theory supports unseen after TRAIN + BALL  valid: {theory_missing_from_observed_b_rows.shape[0]}")

    # Sanity check: generated valid novels should be subset of theory
    if novel_valid_rows_g.shape[0] > 0:
        n_g_valid_in_theory = int(np.isin(novel_valid_void_g, theory_void).sum())
    else:
        n_g_valid_in_theory = 0

    if novel_valid_rows_b.shape[0] > 0:
        n_b_valid_in_theory = int(np.isin(novel_valid_void_b, theory_void).sum())
    else:
        n_b_valid_in_theory = 0

    print(f"[CHECK] GAUSS valid novel supports inside theory: {n_g_valid_in_theory}/{novel_valid_rows_g.shape[0]}")
    print(f"[CHECK] BALL  valid novel supports inside theory: {n_b_valid_in_theory}/{novel_valid_rows_b.shape[0]}")

    # Final summary
    print("\n========== SUMMARY ==========")
    print(f"Radius used for BALL: R = {R:.6f} (p{pctl:.1f} of ||mu(x)||)")
    print(f"THEORY   : total_valid={n_theory}")
    print(f"TRAIN    : unique_supports={n_unique_train_supports}")
    print(f"GAUSSIAN : coverage={cov_g:.2f}% | unique={n_uniq_g} | novel={novel_rows_g.shape[0]} | novel_valid={int(novel_valid_mask_g.sum())} | observed_valid_total={n_observed_valid_g}")
    print(f"BALL     : coverage={cov_b:.2f}% | unique={n_uniq_b} | novel={novel_rows_b.shape[0]} | novel_valid={int(novel_valid_mask_b.sum())} | observed_valid_total={n_observed_valid_b}")
    print("============================\n")

    # Writes the missing lists
    header_g = (f"Missing rows under GAUSSIAN sampling\n"
                f"coverage={cov_g:.2f}% present={n_pres_g}/{n_train}\n"
                f"n_new={args.n_new} seed={args.seed} latent_dim={latent_dim}")
    write_missing(args.missing_gaussian, miss_g, X_phys_train, term_labels, binary_idx, cont_idx, header_g)
    print(f"[GAUSS] Wrote missing rows to: {args.missing_gaussian}")

    header_b = (f"Missing rows under BALL-uniform sampling\n"
                f"coverage={cov_b:.2f}% present={n_pres_b}/{n_train}\n"
                f"n_new={args.n_new} seed={args.seed} latent_dim={latent_dim}\n"
                f"R=p{pctl:.1f}(||mu||)={R:.6f}")
    write_missing(args.missing_ball, miss_b, X_phys_train, term_labels, binary_idx, cont_idx, header_b)
    print(f"[BALL] Wrote missing rows to: {args.missing_ball}")

    # Writes the novel structures
    header_ng = (f"Novel decoded structures under GAUSSIAN sampling\n"
                 f"n_new={args.n_new} seed={args.seed} latent_dim={latent_dim}\n"
                 f"novel_total={novel_rows_g.shape[0]} novel_valid={int(novel_valid_mask_g.sum())}")
    write_novel(args.novel_gaussian, novel_rows_g, novel_valid_mask_g, term_labels,
                binary_idx, cont_idx, classify_family_fn, header_ng)
    print(f"[GAUSS] Wrote novel structures to: {args.novel_gaussian}")

    header_nb = (f"Novel decoded structures under BALL-uniform sampling\n"
                 f"n_new={args.n_new} seed={args.seed} latent_dim={latent_dim}\n"
                 f"R=p{pctl:.1f}(||mu||)={R:.6f}\n"
                 f"novel_total={novel_rows_b.shape[0]} novel_valid={int(novel_valid_mask_b.sum())}")
    write_novel(args.novel_ball, novel_rows_b, novel_valid_mask_b, term_labels,
                binary_idx, cont_idx, classify_family_fn, header_nb)
    print(f"[BALL] Wrote novel structures to: {args.novel_ball}")

    # Optional printing for inspection
    if args.print_missing > 0 and term_labels is not None:
        print(f"\n--- First {args.print_missing} missing (GAUSS) ---")
        for idx in miss_g[:args.print_missing]:
            expr = pde_row_to_string(
                X_phys_train[idx].astype(np.float32),
                term_labels,
                binary_idx=binary_idx,
                cont_idx=cont_idx
            )
            print(f"{idx}: {expr}")

        print(f"\n--- First {args.print_missing} missing (BALL) ---")
        for idx in miss_b[:args.print_missing]:
            expr = pde_row_to_string(
                X_phys_train[idx].astype(np.float32),
                term_labels,
                binary_idx=binary_idx,
                cont_idx=cont_idx
            )
            print(f"{idx}: {expr}")

    if args.print_novel > 0 and term_labels is not None:
        print(f"\n--- First {args.print_novel} novel valid (GAUSS) ---")
        valid_idx_g = np.where(novel_valid_mask_g)[0]
        for i in valid_idx_g[:args.print_novel]:
            expr = pde_row_to_string(
                novel_rows_g[i].astype(np.float32),
                term_labels,
                binary_idx=binary_idx,
                cont_idx=cont_idx
            )
            fam = classify_family_fn(novel_rows_g[i])
            print(f"{i}: [{fam}] {expr}")

        print(f"\n--- First {args.print_novel} novel valid (BALL) ---")
        valid_idx_b = np.where(novel_valid_mask_b)[0]
        for i in valid_idx_b[:args.print_novel]:
            expr = pde_row_to_string(
                novel_rows_b[i].astype(np.float32),
                term_labels,
                binary_idx=binary_idx,
                cont_idx=cont_idx
            )
            fam = classify_family_fn(novel_rows_b[i])
            print(f"{i}: [{fam}] {expr}")

    if term_labels is not None and args.print_theory_examples > 0:
        print_rows_with_expr(
            theory_valid_rows,
            title=f"First {args.print_theory_examples} theoretical valid supports",
            max_n=args.print_theory_examples,
            term_labels=term_labels,
            binary_idx=binary_idx,
            cont_idx=cont_idx,
            classify_family_fn=classify_family_fn
        )

    if term_labels is not None and args.print_theory_missing_from_train > 0:
        print_rows_with_expr(
            theory_missing_from_train_rows,
            title=f"First {args.print_theory_missing_from_train} theoretical valid supports absent from training",
            max_n=args.print_theory_missing_from_train,
            term_labels=term_labels,
            binary_idx=binary_idx,
            cont_idx=cont_idx,
            classify_family_fn=classify_family_fn
        )

    if term_labels is not None and args.print_theory_missing_from_observed > 0:
        print_rows_with_expr(
            theory_missing_from_observed_g_rows,
            title=f"First {args.print_theory_missing_from_observed} theoretical valid supports absent from TRAIN+GAUSS-valid",
            max_n=args.print_theory_missing_from_observed,
            term_labels=term_labels,
            binary_idx=binary_idx,
            cont_idx=cont_idx,
            classify_family_fn=classify_family_fn
        )
        print_rows_with_expr(
            theory_missing_from_observed_b_rows,
            title=f"First {args.print_theory_missing_from_observed} theoretical valid supports absent from TRAIN+BALL-valid",
            max_n=args.print_theory_missing_from_observed,
            term_labels=term_labels,
            binary_idx=binary_idx,
            cont_idx=cont_idx,
            classify_family_fn=classify_family_fn
        )


if __name__ == "__main__":
    main()
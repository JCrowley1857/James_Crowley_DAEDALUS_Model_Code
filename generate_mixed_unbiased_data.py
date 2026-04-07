import numpy as np

from Structural_Rules_Unbiased import sample_pde_configs

# Specification of Output Paths
XMIX_PATH   = "X_mixed_unbiased.npy"
COUNTS_PATH = "counts_active_unbiased.npy"


# Term Library - 67 Terms [Linear & Nonlinear]
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

linear_idx    = list(range(0, 12))
nonlinear_idx = list(range(12, n_terms))

order0 = [0, 1, 2]
order1 = [3, 4, 5]
order2 = [6, 7, 8, 9, 10, 11]
order3 = []

order_map = {}
for i in order0:
    order_map[i] = 0
for i in order1:
    order_map[i] = 1
for i in order2:
    order_map[i] = 2

nonlinear_third_order_idx = np.array([], dtype=int)

# Specification of the first- & second-order terms
first_order_idx  = np.array([3, 4, 5], dtype=int)
second_order_idx = np.array([6, 7, 8, 9, 10, 11], dtype=int)

# Specification of the Binary & Continuous Indexing
cont_idx   = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=int)
binary_idx = np.setdiff1d(np.arange(n_terms), cont_idx)


# Coefficient ranges for second-order diagnonal and off-diagonal terms
DIAG_MIN = 0.2
DIAG_MAX = 1.0
OFFDIAG_MIN = -0.2
OFFDIAG_MAX =  0.2
EIG_TOL = 1e-8


def enforce_min_abs(vals, min_abs=0.01):
    vals = vals.astype(np.float32)
    small = np.abs(vals) < min_abs
    vals[small] = np.sign(vals[small] + 1e-12) * min_abs
    return vals


# Nonlinear dependency indicators
token_to_linear_index = {
    "Ux": 3, "Uy": 4, "Uz": 5,
    "Uxx": 6, "Uyy": 7, "Uzz": 8,
    "Uxy": 9, "Uxz": 10, "Uyz": 11,
}


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


# The list of physics constraints - known as the Physics Plausibility Filter
SECOND_DERIV_TOKENS = ("Uxx", "Uyy", "Uzz", "Uxy", "Uxz", "Uyz")
FIRST_DERIV_TOKENS  = ("Ux", "Uy", "Uz")


def nonlinear_requires_second_order(j: int) -> bool:
    lbl = term_labels[j]
    return any(tok in lbl for tok in SECOND_DERIV_TOKENS)


def nonlinear_requires_first_order(j: int) -> bool:
    lbl = term_labels[j]
    return any(tok in lbl for tok in FIRST_DERIV_TOKENS)


def is_physics_plausible(x: np.ndarray) -> bool:
    has_ut  = (x[0] == 1)
    has_utt = (x[1] == 1)
    has_u   = (x[2] == 1)

    n_first  = int(x[order1].sum())
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

    # family structure logic (coefficient-level PD check is done later)
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

        # strict nonlinear dependency rule
        req_lin = nonlinear_required_linear_indices(j)
        for idx_req in req_lin:
            if x[idx_req] != 1:
                return False

    total_on = int(x.sum())
    if total_on < 2 or total_on > 6:
        return False

    return True


# A_s matrix row determination - for budiling the correct 1x1, 2x2 or 3x3 matrix by rows active
def build_active_spatial_submatrix_from_row(row: np.ndarray, tol: float = EIG_TOL):
    B = np.array([
        [row[6],  row[9],  row[10]],
        [row[9],  row[7],  row[11]],
        [row[10], row[11], row[8]],
    ], dtype=np.float32)

    # active axes
    x_active = (abs(row[6]) > tol) or (abs(row[9]) > tol) or (abs(row[10]) > tol)
    y_active = (abs(row[7]) > tol) or (abs(row[9]) > tol) or (abs(row[11]) > tol)
    z_active = (abs(row[8]) > tol) or (abs(row[10]) > tol) or (abs(row[11]) > tol)

    axes = []
    if x_active:
        axes.append(0)
    if y_active:
        axes.append(1)
    if z_active:
        axes.append(2)

    if len(axes) == 0:
        return None

    return B[np.ix_(axes, axes)]

# Function for determining the positive definitveness of the A_s matrix
def is_positive_definite_active_spatial_submatrix(row: np.ndarray, tol: float = EIG_TOL) -> bool:
    Bs = build_active_spatial_submatrix_from_row(row, tol=tol)
    if Bs is None:
        return False
    eigs = np.linalg.eigvalsh(Bs)
    return bool(np.min(eigs) > tol)


# ------------------------ Main ------------------------
def main():
    print("[DATA] Sampling raw configurations with controlled sparsity...")

    TOTAL_ROWS = 20_000_000

    # Weighted distribution for the number of linear terms active
    k_vals  = np.array([1, 2, 3, 4, 5, 6])
    k_probs = np.array([0.12, 0.30, 0.28, 0.18, 0.08, 0.04])

    rows_per_k = (TOTAL_ROWS * k_probs).astype(int)

    X_parts = []

    for k, n_rows_k in zip(k_vals, rows_per_k):
        print(f"[DATA] Sampling {n_rows_k} rows with {k} linear terms")

        X_k = sample_pde_configs(
            n_rows=n_rows_k,
            n_terms=n_terms,
            linear_idx=linear_idx,
            nonlinear_idx=nonlinear_idx,
            order_map=order_map,
            k_linear_range=(k, k),   # fixed number of linear terms
            k_nonlinear_range=(0, 10),
            k_order_linear=((0, 0), (0, 0), (0, 0), (0, 0))
        )

        X_parts.append(X_k)

    X = np.vstack(X_parts)

    print("[DATA] Combined dataset size:", X.shape)

    # Shuffles so the batches of linear terms are not grouped together, prevents biased learning from dataset
    np.random.shuffle(X)

    print("[DATA] Enforcing binary-structure physics plausibility...")
    mask = np.apply_along_axis(is_physics_plausible, 1, X).astype(bool)
    X_phys = X[mask]
    print(f"[DATA] Kept {X_phys.shape[0]} / {X.shape[0]} rows ({mask.mean():.2%} retained)")

    # Assigns the coefficients
    X_mixed = X_phys.astype(np.float32).copy()

    # first-order coefficients 
    mask1 = X_phys[:, first_order_idx].astype(bool)
    coeffs1 = np.random.uniform(-1.0, 1.0, size=mask1.shape).astype(np.float32)
    coeffs1 = enforce_min_abs(coeffs1, min_abs=0.01)
    X_mixed[:, first_order_idx] = coeffs1 * mask1

    # second-order coefficients
    diag_idx = np.array([6, 7, 8], dtype=int)
    off_idx  = np.array([9, 10, 11], dtype=int)

    mask_diag = X_phys[:, diag_idx].astype(bool)
    diag_vals = np.random.uniform(DIAG_MIN, DIAG_MAX, size=mask_diag.shape).astype(np.float32)
    X_mixed[:, diag_idx] = diag_vals * mask_diag

    mask_off = X_phys[:, off_idx].astype(bool)
    off_vals = np.random.uniform(OFFDIAG_MIN, OFFDIAG_MAX, size=mask_off.shape).astype(np.float32)
    X_mixed[:, off_idx] = off_vals * mask_off

    utt_active = X_phys[:, 1].astype(bool)
    if np.any(utt_active):
        X_mixed[utt_active, 1] = -np.random.uniform(0.1, 1.0, size=int(utt_active.sum())).astype(np.float32)

    # Ut made negative 1.0 if its determined to be a parabolic equation
    parabolic_active = (
        X_phys[:, 0].astype(bool) &
        ~X_phys[:, 1].astype(bool) &
        np.any(X_phys[:, second_order_idx].astype(bool), axis=1)
    )
    if np.any(parabolic_active):
        X_mixed[parabolic_active, 0] = -1.0

    # Positive Definitve matrix filter
    keep = np.ones((X_mixed.shape[0],), dtype=bool)

    for i in range(X_mixed.shape[0]):
        row = X_mixed[i]

        has_ut  = (row[0] == 1)
        has_utt = (abs(row[1]) > EIG_TOL)
        has_spatial_2nd = np.any(np.abs(row[second_order_idx]) > EIG_TOL)

        # first-order families - no A_s matrix to test
        if not has_spatial_2nd:
            continue

        # For second-order equations - check the positive definitiveness of the A_s matrix of coefficients
        if ((not has_ut) and (not has_utt)) or (has_ut and (not has_utt)) or (has_utt and (not has_ut)):
            if not is_positive_definite_active_spatial_submatrix(row, tol=EIG_TOL):
                keep[i] = False

    X_phys  = X_phys[keep]
    X_mixed = X_mixed[keep]

    print(f"[DATA] After active-submatrix PD filter: kept {X_mixed.shape[0]} rows")

    np.save(XMIX_PATH, X_mixed)
    counts_active = X_phys.sum(axis=1).astype(np.int32)
    np.save(COUNTS_PATH, counts_active)

    print(f"[SAVE] X_mixed -> {XMIX_PATH}")
    print(f"[SAVE] counts_active -> {COUNTS_PATH}")
    print("[DONE] Data generation complete.")


if __name__ == "__main__":
    main()
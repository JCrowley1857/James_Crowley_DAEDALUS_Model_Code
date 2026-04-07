import numpy as np

rng = np.random.default_rng(0)

def sample_pde_configs(
    n_rows=600000,
    n_terms=67,
    linear_idx=None,
    nonlinear_idx=None,
    order_map=None,
    k_linear_range=(1, 3),
    k_nonlinear_range=(1, 3),
    k_order_linear=((0,0), (1,2), (0,1), (0,0)),
    max_tries=5000000
):
    linear_idx = linear_idx or list(range(0, 12))
    nonlinear_idx = nonlinear_idx or list(range(12, n_terms))
    order_map = order_map or {}

    linear_by_order = {0:[], 1:[], 2:[], 3:[]}
    for i in linear_idx:
        o = order_map.get(i, 1)
        linear_by_order[o].append(i)

    X = np.zeros((n_rows, n_terms), dtype=np.uint8)
    seen = set()
    i = 0
    tries = 0

    while i < n_rows and tries < max_tries:
        tries += 1
        x = np.zeros(n_terms, dtype=np.uint8)

        for o, krange in zip([0,1,2,3], k_order_linear):
            kmin, kmax = krange
            k = rng.integers(kmin, kmax+1) if kmax >= kmin and len(linear_by_order[o]) > 0 else 0
            if k > 0:
                choose = rng.choice(linear_by_order[o], size=min(k, len(linear_by_order[o])), replace=False)
                x[choose] = 1

        k_lin_min, k_lin_max = k_linear_range
        target_lin = rng.integers(k_lin_min, k_lin_max+1)

        current_lin_idx = np.where(x[linear_idx] == 1)[0]
        current_lin = len(current_lin_idx)

        if current_lin > target_lin:
            to_off = rng.choice(current_lin_idx, size=current_lin - target_lin, replace=False)
            x[np.array(linear_idx)[to_off]] = 0
        elif current_lin < target_lin:
            remaining = [j for j in linear_idx if x[j] == 0]
            add = min(target_lin - current_lin, len(remaining))
            if add > 0:
                picks = rng.choice(remaining, size=add, replace=False)
                x[picks] = 1

        k_nl_min, k_nl_max = k_nonlinear_range
        k_nl = rng.integers(k_nl_min, k_nl_max+1) if k_nl_max >= k_nl_min else 0
        if k_nl > 0 and len(nonlinear_idx) > 0:
            picks = rng.choice(nonlinear_idx, size=min(k_nl, len(nonlinear_idx)), replace=False)
            x[picks] = 1

        key = tuple(x.tolist())
        if key in seen:
            continue
        seen.add(key)
        X[i] = x
        i += 1

    if i < n_rows:
        print(f"Warning: generated only {i} unique rows; consider relaxing rules.")
        X = X[:i]
    return X
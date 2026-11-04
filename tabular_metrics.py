import numpy as np
import torch
import torch.nn.functional as F


def sample_perturbations(X, n_perturb, num_idx, ohe_groups, noise_std=0.2, g=None):
    '''
    Sample valid tabular perturbations for a batch of instances.
    Numeric (and single-column binary) features get Gaussian noise with
    std = noise_std * feature std, every OHE group switches to a random
    different category. Returns delta (m, n_perturb, f) such that the
    perturbed input is x - delta.
    '''
    m, f = X.shape
    delta = torch.zeros(m, n_perturb, f, dtype=X.dtype)
    if len(num_idx) > 0:
        sigma = X[:, num_idx].std(dim=0)
        sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
        noise = torch.randn(m, n_perturb, len(num_idx), generator=g, dtype=X.dtype)
        delta[:, :, num_idx] = noise * (noise_std * sigma)
    for cols in ohe_groups:
        k = len(cols)
        orig = X[:, cols]                                   # (m, k), one-hot
        orig_cat = orig.argmax(dim=1)                       # (m,)
        r = torch.randint(0, k, (m, n_perturb), generator=g)
        clash = r == orig_cat.unsqueeze(1)
        while clash.any():                                  # resample until category changes
            new = torch.randint(0, k, clash.shape, generator=g)
            r[clash] = new[clash]
            clash = r == orig_cat.unsqueeze(1)
        onehot = F.one_hot(r, k).to(X.dtype)                # (m, n_perturb, k)
        delta[:, :, cols] = orig.unsqueeze(1) - onehot
    return delta


def infidelity(X, explanations, model, delta):
    '''
    Infidelity (Yeh et al., 2019) with per-instance optimal scaling:
        infid = min_beta E_delta[(f(x) - f(x - delta) - beta * <phi(x), delta>)^2]
    Lower is better. X: (m, f), explanations: (m, f), delta: (m, P, f).
    '''
    m, P, f = delta.shape
    x_pert = (X.unsqueeze(1) - delta).reshape(m * P, f)
    with torch.no_grad():
        fx = model(X).squeeze(-1)
        fx_pert = model(x_pert).squeeze(-1).reshape(m, P)
    I = fx.unsqueeze(1) - fx_pert                                   # (m, P)
    S = (explanations.unsqueeze(1) * delta).sum(dim=-1)             # (m, P)
    num = (I * S).sum(dim=1)
    den = (S * S).sum(dim=1)
    beta = torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.zeros_like(num))
    return ((I - beta.unsqueeze(1) * S) ** 2).mean(dim=1).mean().item()


def _attribution_matrix(attributions, m, f, detach=True):
    '''
    Normalize an explainer output to a dense (m, f) attribution matrix.
    Handles tuple outputs and extra dimensions like (m, 1, f). detach=False
    keeps the autograd graph to theta (required when the matrix feeds the CRMO
    losses); the default detaches for pure evaluation.
    '''
    if isinstance(attributions, tuple):
        attributions = attributions[0]
    if detach:
        attributions = attributions.detach()
    if attributions.dim() == 1:
        attributions = attributions.reshape(1, -1)
    elif attributions.dim() > 2:
        attributions = attributions.reshape(attributions.shape[0], -1)
    if tuple(attributions.shape) != (m, f):
        raise ValueError(
            f"explain_fn returned shape {tuple(attributions.shape)}, expected {(m, f)}. "
            "The explainer must yield one attribution per input feature, e.g. "
            "Saliency, InputXGradient, IntegratedGradients, or LayerGradientXActivation "
            "with attribute_to_layer_input=True. LayerGradCam sums over the layer's "
            "channels and cannot produce per-feature attributions here."
        )
    return attributions


def max_sensitivity(X, explanations, explain_fn, delta):
    '''
    Maximum sensitivity (Yeh et al., 2019):
        max over perturbations of ||phi(x) - phi(x - delta)||_2 / ||phi(x)||_2,
        averaged over instances. Lower is better.
    '''
    m, P, f = delta.shape
    x_pert = (X.unsqueeze(1) - delta).reshape(m * P, f)
    E_pert = _attribution_matrix(explain_fn(x_pert), m * P, f).reshape(m, P, f)
    diff = (E_pert - explanations.unsqueeze(1)).norm(dim=-1)        # (m, P)
    base = explanations.norm(dim=-1).clamp_min(1e-12)               # (m,)
    return (diff / base.unsqueeze(1)).max(dim=1).values.mean().item()


def faithfulness_correlation(X, explanations, model, num_idx, ohe_groups,
                             n_subsets=50, baseline=0.0, rs=None, g=None):
    '''
    Faithfulness correlation (Bhatt et al., 2020), removal-based: random subsets of
    feature units (single numeric/binary columns and whole OHE groups) are replaced
    by a baseline value; the Pearson correlation between the removed units'
    attribution mass and the induced drop in the model score is computed per
    instance and averaged. Higher is better.
    '''
    m, f = X.shape
    units = [[c] for c in num_idx] + [list(c) for c in ohe_groups]
    U = len(units)
    unit_of_col = torch.zeros(f, dtype=torch.long)
    for u, cols in enumerate(units):
        unit_of_col[cols] = u
    unit_phi = torch.stack([explanations[:, cols].sum(dim=-1) for cols in units], dim=1)  # (m, U)
    with torch.no_grad():
        fx = model(X).squeeze(-1)
    phis, dys = [], []
    for _ in range(n_subsets):
        k = rs.randint(1, U + 1)                                    # random subset size
        perm = torch.rand(m, U, generator=g).argsort(dim=1)
        mask = torch.zeros(m, U, dtype=torch.bool)
        mask.scatter_(1, perm[:, :k], True)
        col_mask = mask[:, unit_of_col]                             # (m, f)
        Xp = X.clone()
        Xp[col_mask] = baseline
        with torch.no_grad():
            fp = model(Xp).squeeze(-1)
        phis.append((unit_phi * mask).sum(dim=1))
        dys.append(fx - fp)
    phi = torch.stack(phis, dim=1)                                  # (m, n_subsets)
    dy = torch.stack(dys, dim=1)
    phi_c = phi - phi.mean(dim=1, keepdim=True)
    dy_c = dy - dy.mean(dim=1, keepdim=True)
    num_ = (phi_c * dy_c).sum(dim=1)
    den_ = phi_c.norm(dim=1) * dy_c.norm(dim=1)
    r = torch.where(den_ > 1e-12, num_ / den_.clamp_min(1e-12),
                    torch.full_like(num_, float('nan')))
    return float(torch.nanmean(r).item())


def _spec_name(spec):
    '''
    Human-readable key for an explainer spec.
    '''
    if isinstance(spec, str):
        return spec
    if hasattr(spec, "attribute"):
        return type(spec).__name__
    return getattr(spec, "__name__", "callable")


def _resolve_explainer(spec, model):
    '''
    Normalize an explainer spec to a callable explain(X) -> (n, f) that also runs
    under an outer torch.no_grad(): callables and Captum .attribute objects are
    wrapped in torch.enable_grad (gradient surrogates need grad even when the
    caller disables it), and name strings are resolved via crmo.make_explainer
    (lazy import, since crmo imports this module).
    '''
    if isinstance(spec, str):
        from crmo import make_explainer  # lazy import to avoid a circular dependency
        fn = make_explainer(spec, model)
    elif hasattr(spec, "attribute"):
        fn = lambda Z: spec.attribute(Z)
    elif callable(spec):
        fn = spec
    else:
        raise TypeError(f"Unsupported explainer spec: {spec!r}")

    def explain(X):
        with torch.enable_grad():
            return fn(X)
    return explain


def evaluate_exp_quality(X, model, explain_fn, num_idx, ohe_groups,
                         noise_std=0.2, subset_n=100, n_perturb=100,
                         n_subsets=50, seed=None):
    '''
    Compute input-space explanation quality metrics on a random subset of X:
    Infidelity and MaxSensitivity (lower is better) and Faithfulness (higher
    is better). explain_fn may be a callable, a Captum object with .attribute,
    or a name resolved via crmo.make_explainer ("grad", "ixg", "smoothgrad",
    "ig", "gxa"). Returns a flat dict of the three metrics.
    '''
    fn = _resolve_explainer(explain_fn, model)
    rs = np.random.RandomState(seed) if seed is not None else np.random
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    was_training = model.training
    model.eval()
    try:
        idx = rs.choice(len(X), size=min(subset_n, len(X)), replace=False)
        X_sub = X[idx]
        delta = sample_perturbations(X_sub, n_perturb, num_idx, ohe_groups,
                                     noise_std=noise_std, g=g)
        E_sub = _attribution_matrix(fn(X_sub), X_sub.shape[0], X_sub.shape[1])
        return {
            "Infidelity": infidelity(X_sub, E_sub, model, delta),
            "MaxSensitivity": max_sensitivity(X_sub, E_sub, fn, delta),
            "Faithfulness": faithfulness_correlation(X_sub, E_sub, model, num_idx,
                                                     ohe_groups, n_subsets=n_subsets,
                                                     rs=rs, g=g),
        }
    finally:
        if was_training:
            model.train()
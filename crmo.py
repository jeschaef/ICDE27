import numpy as np
import torch
import tqdm

from tabular_metrics import _attribution_matrix, sample_perturbations, _resolve_explainer


def _euclidean(a, b):
    '''
    Default differentiable distance, valid for inputs and explanations alike.
    '''
    return (a - b).norm(dim=-1)


def _predict_classes(out):
    '''
    Class predictions for binary (single logit/probability) or multi-class outputs.
    '''
    if out.dim() > 1 and out.shape[1] > 1:
        return out.argmax(dim=1)
    return (out.reshape(-1) >= 0.5).long()


def grad_x_input(model):
    '''
    Differentiable input-gradient explainer: a trainable surrogate for Saliency /
    IntegratedGradients. Attributions g(x) * x are computed with create_graph=True
    so the second-order path back to theta is kept. Designed for single-output
    (binary) models; for multi-class, pass a layer-based explainer or select the
    class logit before summing. Layer-based explainers (LayerGradCam,
    LayerGradientXActivation with multiply_by_inputs=True) are differentiable
    through their activation term and do not need this wrapper.
    '''
    def explain(X):
        X = X.clone().requires_grad_(True)
        with torch.enable_grad():  # also runs under an outer torch.no_grad()
            out = model(X)
            g = torch.autograd.grad(out.sum(), X, create_graph=True)[0]
        return g * X
    return explain


def saliency(model):
    '''
    Differentiable Saliency (plain input gradients, "Grad"). create_graph=True
    keeps the second-order path to theta.
    '''
    def explain(X):
        X = X.clone().requires_grad_(True)
        with torch.enable_grad():
            out = model(X)
            return torch.autograd.grad(out.sum(), X, create_graph=True)[0]

    return explain


def smooth_grad(model, n_samples=5, noise_std=0.1):
    '''
    Differentiable SmoothGrad: input gradients averaged over n_samples noisy
    copies of the input. The noise is constant w.r.t. theta; create_graph=True
    keeps the second-order path. Reproducibility via global torch.manual_seed.
    '''

    def explain(X):
        X = X.clone().requires_grad_(True)
        total = torch.zeros_like(X)
        with torch.enable_grad():
            for _ in range(n_samples):
                noise = noise_std * torch.randn(X.shape, dtype=X.dtype, device=X.device)
                out = model(X + noise)
                total = total + torch.autograd.grad(out.sum(), X, create_graph=True)[0]
        return total / n_samples

    return explain


def integrated_gradients(model, n_steps=20, baseline=None):
    '''
    Differentiable Integrated Gradients (Riemann approximation):
    (x - x0) * mean gradient along the interpolated path, create_graph=True.
    Costs n_steps forward passes per explain() call -- keep n_steps small
    (8-20) when used inside the finetuning loss.
    '''

    def explain(X):
        X = X.clone().requires_grad_(True)
        x0 = torch.zeros_like(X) if baseline is None else \
            torch.as_tensor(baseline, dtype=X.dtype, device=X.device).expand_as(X)
        total = torch.zeros_like(X)
        with torch.enable_grad():
            for a in range(1, n_steps + 1):
                zi = (x0 + (a / n_steps) * (X - x0)).detach().requires_grad_(True)
                out = model(zi)
                total = total + torch.autograd.grad(out.sum(), zi, create_graph=True)[0]
        return (X - x0) * (total / n_steps)
    return explain


def smooth_grad_sq(model, n_samples=5, noise_std=0.1):
    '''
    SmoothGrad Squared: mean of SQUARED input gradients over noisy copies.
    Sharper, sparser maps than SmoothGrad. Noise is constant w.r.t. theta;
    create_graph=True keeps the second-order path.
    '''
    def explain(X):
        X = X.clone().requires_grad_(True)
        total = torch.zeros_like(X)
        with torch.enable_grad():
            for _ in range(n_samples):
                noise = noise_std * torch.randn(X.shape, dtype=X.dtype, device=X.device)
                out = model(X + noise)
                total = total + torch.autograd.grad(out.sum(), X, create_graph=True)[0] ** 2
        return total / n_samples
    return explain


def var_grad(model, n_samples=5, noise_std=0.1):
    '''
    VarGrad: variance of the input gradients over noisy copies,
    E[g^2] - E[g]^2. Noise is constant w.r.t. theta; create_graph=True
    keeps the second-order path. Needs enough noise to produce non-zero
    variance (with tiny noise_std it collapses to ~0).
    '''
    def explain(X):
        X = X.clone().requires_grad_(True)
        total = torch.zeros_like(X)
        total_sq = torch.zeros_like(X)
        with torch.enable_grad():
            for _ in range(n_samples):
                noise = noise_std * torch.randn(X.shape, dtype=X.dtype, device=X.device)
                out = model(X + noise)
                g = torch.autograd.grad(out.sum(), X, create_graph=True)[0]
                total = total + g
                total_sq = total_sq + g ** 2
        mean = total / n_samples
        return total_sq / n_samples - mean ** 2
    return explain


def gradient_shap(model, n_samples=20, stdev=0.01, seed=None):
    '''
    GradientShap surrogate (Captum-compatible, zeros baseline): Gaussian noise
    on the input, gradients at random points along the 0 -> x path, averaged,
    multiplied by x. The noise and path points are detached (constant w.r.t.
    theta); create_graph=True keeps the second-order path.
    '''
    gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def explain(X):
        X = X.clone().requires_grad_(True)
        total = torch.zeros_like(X)
        with torch.enable_grad():
            for _ in range(n_samples):
                alpha = torch.rand(X.shape[0], 1, generator=gen, dtype=X.dtype,
                                   device=X.device)
                noise = stdev * torch.randn(X.shape, generator=gen, dtype=X.dtype,
                                            device=X.device)
                zi = (alpha * X + noise).detach().requires_grad_(True)
                out = model(zi)
                total = total + torch.autograd.grad(out.sum(), zi, create_graph=True)[0]
        return (total / n_samples) * X
    return explain


def guided_backprop(model):
    '''
    Differentiable GuidedBackprop surrogate: ReLU backward hooks pass only the
    positive gradient components (masked by a positive forward input), computed
    with create_graph=True so the regularizers stay attached to theta. The
    hooks are registered per call and removed afterwards. Caveat: the modified
    gradient is not the gradient of a scalar function, so its second-order
    meaning is heuristic; the step-0 audit still verifies the theta path.
    '''
    relus = [m for m in model.modules() if isinstance(m, torch.nn.ReLU)]
    if not relus:
        raise ValueError("guided_backprop requires ReLU activations.")

    def explain(X):
        fwd_handles, bwd_handles = [], []
        def fwd_hook(mod, inp, out):
            mod._gbp_input = inp[0]
        def bwd_hook(mod, gin, gout):
            mask = (mod._gbp_input > 0).to(gout[0].dtype)
            return (gout[0].clamp(min=0.0) * mask,) + tuple(gin[1:])
        try:
            for r in relus:
                fwd_handles.append(r.register_forward_hook(fwd_hook))
                bwd_handles.append(r.register_full_backward_hook(bwd_hook))
            X = X.clone().requires_grad_(True)
            with torch.enable_grad():
                out = model(X)
                return torch.autograd.grad(out.sum(), X, create_graph=True)[0]
        finally:
            for h in fwd_handles + bwd_handles:
                h.remove()
    return explain


def layer_integrated_gradients(model, layer, n_steps=20):
    '''
    LayerIntegratedGradients attributed to the layer INPUT: the layer analog of
    Integrated Gradients; with the first layer it equals IG per input feature.
    Requires a Captum version whose layer methods forward grad_kwargs to
    compute_layer_gradients_and_eval (verify with the step-0 audit).
    '''
    from captum.attr import LayerIntegratedGradients
    E = LayerIntegratedGradients(model, layer)
    return lambda X: E.attribute(X, attribute_to_layer_input=True,
                                 n_steps=n_steps,
                                 grad_kwargs={"create_graph": True})


def layer_gxa(model, layer):
    '''
    Captum LayerGradientXActivation attributed to the layer INPUT: differentiable
    through the activation term, no create_graph needed. Pass the FIRST layer so
    the layer input is the model input and attributions have one value per input
    feature. (GradCam-style channel-summed variants collapse to one scalar per
    instance and cannot serve here -- the shape guard rejects them.)
    '''
    from captum.attr import LayerGradientXActivation
    E = LayerGradientXActivation(model, layer)
    # create_graph keeps the second-order graph: without it the layer
    # gradients are detached constants and the attributions carry no
    # gradient back to theta.
    return lambda X: E.attribute(X, attribute_to_layer_input=True,
                                 grad_kwargs={"create_graph": True})


def _first_layer(model, layer=None):
    if layer is not None:
        return layer
    net = getattr(model, "net", None)
    if net is not None and len(net) > 0:
        return net[0]
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            return m
    raise ValueError("Could not infer the first layer; pass layer explicitly.")



def make_explainer(spec, model, layer=None):
    '''
    Normalize an explainer spec to a callable explain(X) -> (n, f). Only
    differentiable surrogates are provided, since L_stab/L_robu must stay
    connected to theta (LIME / SHAP / Anchor / ALE-style explainers cannot
    enter the finetuning loss):

        callable                       used as-is
        Captum object with .attribute  wrapped
        "grad" / "saliency"            Saliency (plain input gradients)
        "ixg" / "grad_x_input"         grad x input
        "smoothgrad"                   SmoothGrad (n_samples=5)
        "smoothgrad_sq"                SmoothGrad Squared (mean squared gradients)
        "vargrad"                      VarGrad (gradient variance)
        "ig" / "integrated_gradients"  Integrated Gradients (n_steps=20)
        "gradshap"                     GradientShap (stochastic, zeros baseline)
        "guided" / "gbp"               GuidedBackprop (differentiable hooks)
        "gxa" / "layer_gxa"            LayerGradientXActivation on `layer`
                                       (first layer inferred if not given)
        "lig" / "layer_integrated_gradients"
                                       LayerIntegratedGradients on `layer`
                                       (first layer inferred if not given)

    The gradient-based surrogates differentiate the summed output: they assume
    a single-output (binary) model; for multi-class, select the class logit or
    use "gxa".
    '''
    if hasattr(spec, "attribute"):
        return lambda Z: spec.attribute(Z)
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"Unsupported explainer spec: {spec!r}")
    name = spec.lower()
    if name in ("grad", "saliency"):
        return saliency(model)
    if name in ("ixg", "grad_x_input"):
        return grad_x_input(model)
    if name == "smoothgrad":
        return smooth_grad(model)
    if name in ("smoothgrad_sq", "smoothgradsq"):
        return smooth_grad_sq(model)
    if name in ("vargrad", "var_grad"):
        return var_grad(model)
    if name in ("ig", "integrated_gradients"):
        return integrated_gradients(model)
    if name in ("gradshap", "gradient_shap"):
        return gradient_shap(model)
    if name in ("guided", "guided_backprop", "gbp"):
        return guided_backprop(model)
    if name in ("gxa", "layer_gxa"):
        return layer_gxa(model, _first_layer(model, layer))
    if name in ("lig", "layer_integrated_gradients"):
        return layer_integrated_gradients(model, _first_layer(model, layer))
    raise ValueError(f"Unknown explainer name '{spec}'. Use grad/saliency, ixg/grad_x_input, "
                     "smoothgrad, smoothgrad_sq, vargrad, ig/integrated_gradients, gradshap, "
                     "guided/gbp, gxa/layer_gxa, lig/layer_integrated_gradients, "
                     "a Captum object or a callable.")


def explainer_name(spec):
    if not isinstance(spec, str):
        return type(spec).__name__
    return spec.lower()


def compute_maxv(A, maxv_quantile=0.95, p=2.0):
    pdm = torch.cdist(A, A, p=p)
    iu = torch.triu_indices(len(A), len(A), offset=1)
    return float(pdm[iu[0], iu[1]].quantile(maxv_quantile).clamp_min(1e-12))


def CRMO(model, X, y, explainer, criterion, num_idx, ohe_groups,
             d_D=None, d_E=None,
             delta=0.2, lambda_s=1.0, lambda_r=1.0,
             m=64, k=100, batch_size=None,
             epochs=3,
             layer=None,
             pair_selection="prediction_preserving",
             refresh_pairs_every=1,
             optimizer=None, lr=1e-5,
             eval_fn=None, valid_patience=20, log_every=10,
             seed=None, verbose=True):
    '''
    Consistency-regularized model optimization (joint finetuning):

        J(theta) = L_task(theta)                on mini-batch B
                 + lambda_s * L_stab(theta)     structural regularization
                 + lambda_r * L_robu(theta)     local regularization

    L_stab: explanation similarities of m sampled pairs must match the
            input-space similarities (squared discrepancy on a shared scale).
    L_robu: normalized squared explanation distances between each instance and
            its prediction-preserving perturbations of radius delta.


    Mapping to the pseudocode:
        X, y                  Dataset D
        model, explainer      M_theta and E_theta; pass the pretrained model to
                              finetune it, a fresh model initializes theta.
                              explainer: a single spec normalized by
                              make_explainer: a callable, a Captum object with
                              .attribute, or a name ("grad", "ixg",
                              "smoothgrad", "ig", "gxa"). Only differentiable
                              surrogates can enter the loss.
        d_D, d_E              input / explanation distance (default: euclidean)
        delta                 perturbation radius (std = delta * feature std)
        lambda_s, lambda_r    regularization weights
        layer                 layer for the "gxa" spec (first layer inferred
                              if not given)
        m, k                  pair batch size / total perturbations per instance
        pair_selection        "prediction_preserving" keeps only perturbations
                              that do not flip the predicted class; "all" keeps
                              every candidate
        refresh_pairs_every   full-batch mode only: cache the (model-independent)
                              perturbation deltas for this many steps; 1 = fresh
        eval_fn               optional callable model -> dict, added to results

    criterion must match the model output: BCEWithLogitsLoss for logits,
    BCELoss for probabilities, CrossEntropyLoss for multi-class (y: long).

    Returns (model, results) with results["history"], maxv, config, final
    accuracy / perturbation validity and, if given, results["eval"].
    '''
    rs = np.random.RandomState(seed)
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    d_D = d_D if d_D is not None else _euclidean
    d_E = d_E if d_E is not None else _euclidean

    # E_theta: normalize the explainer spec to a differentiable callable
    explain_fn = make_explainer(explainer, model, layer=layer)

    n, f = X.shape
    full_batch = batch_size is None or batch_size >= n
    batch_size = n if full_batch else batch_size

    model.train()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, cached_delta, stopped = [], None, False
    no_valid_steps, step = 0, 0

    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        perm = torch.from_numpy(rs.permutation(n))
        for start in range(0, n, batch_size):
            bidx = perm[start:start + batch_size]
            X_B, y_B = X[bidx], y[bidx]
            b = len(bidx)

            # ---- Sample m instance pairs P (from B) for structural reg. ----
            ii = torch.from_numpy(rs.randint(0, b, size=m))
            jj = torch.from_numpy(rs.randint(0, b, size=m))
            keep = ii != jj
            ii, jj = ii[keep], jj[keep]

            # ---- Sample / refresh the k candidate perturbations ----
            if not (full_batch and refresh_pairs_every > 1 and cached_delta is not None
                    and step % refresh_pairs_every != 0):
                cached_delta = sample_perturbations(X_B, k, num_idx, ohe_groups,
                                                    noise_std=delta, g=g)
            x_pert_all = (X_B.unsqueeze(1) - cached_delta).reshape(-1, f)

            # ---- Select valid prediction-preserving pairs Q ----
            model.eval()
            with torch.no_grad():
                cls_orig = _predict_classes(model(X_B))
                cls_pert = _predict_classes(model(x_pert_all))
            model.train()
            if pair_selection == "prediction_preserving":
                q_mask = cls_pert == cls_orig.repeat_interleave(k)
            else:  # "all"
                q_mask = torch.ones_like(cls_pert, dtype=torch.bool)
            valid_rate = float(q_mask.float().mean())

            # ---- Task loss on B ----
            loss_task = criterion(model(X_B), y_B)

            # ---- Explanations and regularizers ----
            # The eval metrics are computed on the deterministic network;
            # measuring the regularizers through train-mode dropout noise
            # optimizes a different quantity. Gradients still flow.
            model.eval()
            E_B = _attribution_matrix(explain_fn(X_B), b, f, detach=False)

            # L_stab: similarity discrepancy on the pairs in P
            loss_stab = torch.zeros((), dtype=X.dtype, device=X.device)
            if len(ii) > 0:
                d_in = d_D(X_B[ii], X_B[jj]).detach()  # input-space target
                d_ex = d_E(E_B[ii], E_B[jj])  # explanation space
                # Each vector is normalized by its own batch max, so L_stab
                # shapes only the RELATIVE distance profile. The previous
                # global maxv/maxv_X normalization coupled the absolute
                # explanation scale to the input scale, letting L_stab
                # inflate perturbation distances while the (rank-based)
                # stability metric still improved.
                w_D = 1 - d_in / d_in.max().clamp_min(1e-12)
                w_E = 1 - d_ex / d_ex.max().clamp_min(1e-12)
                loss_stab = ((w_E - w_D) ** 2).mean()

            # L_robu: normalized squared explanation distances on Q
            loss_rob = torch.zeros((), dtype=X.dtype, device=X.device)
            if q_mask.any():
                x_pert = x_pert_all[q_mask]
                E_pert = _attribution_matrix(explain_fn(x_pert), len(x_pert), f, detach=False)
                E_orig = E_B[torch.nonzero(q_mask, as_tuple=True)[0] // k]
                # loss_rob = ((d_E(E_orig, E_pert) / maxv) ** 2).mean()
                loss_rob = (d_E(E_orig, E_pert) ** 2).mean()
            else:
                no_valid_steps += 1
            model.train()

            # ---- Audit + joint objective ----
            if step == 0 and verbose:
                if not (loss_stab.requires_grad or loss_rob.requires_grad):
                    print("WARNING: both regularizers are detached from theta "
                          "(the explainer returns detached attributions); crmo "
                          "will effectively minimize only the task loss.")
                else:
                    # requires_grad alone is not conclusive: attributions can
                    # connect to the input while carrying no gradient to theta.
                    # Probe the parameters once (step 0 only).
                    params = [p for p in model.parameters() if p.requires_grad]
                    grads = torch.autograd.grad(loss_stab + loss_rob, params,
                                                allow_unused=True,
                                                retain_graph=True) if params else ()
                    if len(grads) > 0 and all(g is None for g in grads):
                        print("WARNING: regularizers reach the model input but "
                              "not theta (the explainer's gradients are not "
                              "kept, e.g. layer grads without create_graph); "
                              "crmo will effectively minimize only the task loss.")
            J = loss_task + lambda_s * loss_stab + lambda_r * loss_rob

            optimizer.zero_grad()
            J.backward()
            optimizer.step()

            # ---- Monitor prediction quality and perturbation validity ----
            with torch.no_grad():
                acc = float((cls_orig == y_B.reshape(-1)).float().mean())
            history.append({"epoch": epoch, "step": step,
                            "task": float(loss_task), "stab": float(loss_stab),
                            "rob": float(loss_rob), "J": float(J),
                            "acc": acc, "valid_rate": valid_rate})
            if step % log_every == 0:
                pbar.set_description(f"Epoch {epoch+1}/{epochs} (step {step+1}) | J={float(J):.4f} | "
                      f"Task={float(loss_task):.4f} | Stab={float(loss_stab):.4f} | "
                      f"Robu={float(loss_rob):.4f} | Accuracy={acc:.4f} | Valid Rate={valid_rate:.4f} ")
            step += 1

            if valid_rate == 0.0 and no_valid_steps >= valid_patience:
                if verbose:
                    print(f"crmo: no prediction-preserving perturbations for "
                          f"{valid_patience} steps -- stopping early.")
                stopped = True
            elif valid_rate > 0.0:
                no_valid_steps = 0
            if stopped:
                break
        if stopped:
            break

        # ---- Store evaluation results ----
        results = {"history": history, "steps": step,
                   "explainer": explainer_name(explainer),
                   "delta": delta, "lambda_s": lambda_s, "lambda_r": lambda_r,
                   "final_acc": history[-1]["acc"],
                   "final_valid_rate": history[-1]["valid_rate"]}

    # ---- Return the optimized model and evaluation results ----
    model.eval()
    if eval_fn is not None:
        results["eval"] = eval_fn(model)
    return model, results
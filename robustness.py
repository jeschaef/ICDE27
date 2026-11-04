from typing import Callable

import numpy as np

from crmo import make_explainer, _predict_classes, _euclidean, compute_maxv
from tabular_metrics import sample_perturbations, _attribution_matrix
from util import graph_explain_all, graph_explain_single, get_ohe_columns, match_columns
import scipy
import torch

from GraphXAI.graphxai.datasets import ShapeGGen
from GraphXAI.graphxai.explainers import PGExplainer
from GraphXAI.graphxai.utils.perturb import perturb_node_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_noise(x, std, samples):
    '''
    Add gaussiau noise to a sample.
    '''
    noise = np.random.normal(scale=std, size=(samples, len(x)))
    noise += np.array([x.detach()])
    return noise


def snr_to_std(X, snr):
    '''
    Compute the standard deviation of the data for the given snr.
    '''
    sp = (X ** 2).mean(axis=0)  # Signal Power
    return (sp / snr) ** 0.5  # Std of noise


def lime_exp_to_array(e, label):
    '''
    Transform a lime explanation to a numpy array.
    '''
    elist = e.as_map()[label]
    elist.sort()
    return np.array([val for i, val in elist])


def lime_explain_all(X, model, E, labels):
    '''
    Explain all instances with the lime tabular explainer.
    '''
    f = lambda x: E.explain_instance(x, model.predict_proba, labels=labels)
    return np.array([lime_exp_to_array(f(x), labels[0]) for x in X])


def sample(x, y, model, std_n, samples=10):
    '''
    Sample instances using gaussian noise (numeric data only).
    '''
    pert_x = gaussian_noise(x, std_n, 2 * samples)  # Perturbate double number of samples (binary clf)
    pert_y = model.predict(pert_x)
    return pert_x[pert_y == y][:samples]  # Filter perturbed data by same prediction as x


def sample_torch(x, y, model, std_n, samples=10):
    pert_x = gaussian_noise(x, std_n, 2 * samples)  # Perturbate double number of samples (binary clf)
    pert_x = torch.tensor(pert_x)  # , device=device)
    with torch.no_grad():
        pert_y = model(pert_x).squeeze()
    pert_y = (pert_y >= 0.5).float()
    return pert_x[pert_y == y][:samples]  # Filter perturbed data by same prediction as x


def sample_mixed(x, y, model, std_n, columns, ohe_columns, samples=10):
    '''
    Sample instances from mixed data.
    '''
    n = len(x)
    ohe_idx = get_ohe_columns(ohe_columns, columns)
    num_idx = [i for i in range(n) if i not in ohe_idx]

    # Sample numeric data by adding gaussian noise
    pert_x_num = np.zeros((samples, n))
    pert_x_num[:, num_idx] = gaussian_noise(x[num_idx], std_n, samples)
    pert_x_num[:, ohe_idx] = x[ohe_idx]  # Keep categorical features
    pert_y_num = model.predict(pert_x_num)
    pert_x_num = pert_x_num[pert_y_num == y]  # Filter perturbed data by same prediction as x

    # Sample categorical data by permuting one ohe-feature randomly
    pert_x_cat = np.zeros((samples, n))
    pert_x_cat[:, :] = x
    match = match_columns(ohe_columns, columns)
    match_ohe = {k: v for k, v in match.items() if len(v) > 1}
    for i in range(samples):
        rand_idx = match_ohe[np.random.choice(list(match_ohe))]  # Randomly choose one ohe-feature to permute
        pert_x_cat[i, rand_idx] = shuffle_ohe(x[rand_idx])
    pert_y_cat = model.predict(pert_x_cat)
    pert_x_cat = pert_x_cat[pert_y_cat == y]

    pert_x = np.concatenate((pert_x_cat, pert_x_num))
    np.random.shuffle(pert_x)
    return pert_x[:samples]


def sample_mixed_torch(x, y, model, std_n, columns, ohe_columns, samples=10):
    '''
    Sample instances from mixed data.
    '''
    n = len(x)
    ohe_idx = get_ohe_columns(ohe_columns, columns)
    num_idx = [i for i in range(n) if i not in ohe_idx]

    # Sample numeric data by adding gaussian noise
    pert_x_num = torch.zeros((samples, n))
    pert_x_num[:, num_idx] = torch.tensor(gaussian_noise(x[num_idx], std_n, samples))
    pert_x_num[:, ohe_idx] = x[ohe_idx]  # Keep categorical features
    with torch.no_grad():
        pert_y_num = model.predict(pert_x_num).squeeze()
    pert_y_num = (pert_y_num >= 0.5).float()
    pert_x_num = pert_x_num[pert_y_num == y]  # Filter perturbed data by same prediction as x

    # Sample categorical data by permuting one ohe-feature randomly
    pert_x_cat = torch.zeros((samples, n))
    pert_x_cat[:, :] = x
    match = match_columns(ohe_columns, columns)
    match_ohe = {k: v for k, v in match.items() if len(v) > 1}
    for i in range(samples):
        rand_idx = match_ohe[np.random.choice(list(match_ohe))]  # Randomly choose one ohe-feature to permute
        pert_x_cat[i, rand_idx] = shuffle_ohe_torch(x[rand_idx])
    pert_y_cat = model.predict(pert_x_cat).squeeze()
    pert_y_cat = (pert_y_cat >= 0.5).float()
    pert_x_cat = pert_x_cat[pert_y_cat == y]

    pert_x = torch.concat((pert_x_cat, pert_x_num))
    idx = torch.randperm(pert_x.shape[0])[:samples]
    return pert_x[idx]


def shuffle_ohe(ohe):
    '''
    Return a shuffled copy of a given ohe vector.
    '''
    ohe_shuff = ohe.copy()
    while True:
        np.random.shuffle(ohe_shuff)
        if (ohe_shuff != ohe).any():
            break
    return ohe_shuff


def shuffle_ohe_torch(ohe):
    '''
    Return a shuffled copy of a given ohe vector.
    '''
    ohe_shuff = ohe.clone()
    while True:
        idx = torch.randperm(ohe.shape[0])
        ohe_shuff = ohe_shuff[idx]
        if (ohe_shuff != ohe).any():
            break
    return ohe_shuff


def agg_exp_metric(explanations, y_pred, metric, aggregate=np.max):
    '''
    Aggregate the explanation metric over a list of explanations.
    '''
    e0 = []
    e1 = []
    for i, (e, p) in enumerate(zip(explanations, y_pred)):
        if p == 1:
            e1.append(e)
        else:
            e0.append(e)

    n0 = len(e0)
    n1 = len(e1)
    D = np.zeros((n0, n1))
    for i in range(n0):
        for j in range(n1):
            D[i, j] = metric(explanations[i], explanations[j]).detach()
    return aggregate(D)


def anchor_explain_all(X, E, **params):
    '''
    Explain all instances using Anchor tabular explainer.
    '''
    return [E.explain(x, **params) for x in X]


def transform_anchor(a):
    a_trans = []
    for x in a:
        if ' = ' in x:
            x = x.replace(' = ', ' == ')
            val = x.split(' == ')[1]
            x = x.replace(val, f"'{val}'")
        a_trans.append(x)
    return a_trans


def eval_anchor(a, X):
    '''
    Evaluate an anchor on a dataset.
    '''
    a = transform_anchor(a)
    expr = ' and '.join(a)
    return X.query(expr)


def jaccard_similarity(list1, list2):
    '''
    Jaccard similarity of two lists.
    '''
    s1 = set(list1)
    s2 = set(list2)
    inters = len(s1.intersection(s2))
    if inters == 0:
        return 0
    return float(inters / (len(s1) + len(s2) - inters))


def sorensen_dice(list1, list2):
    '''
    Sorensen-Dice index of two lists.
    '''
    s1 = set(list1)
    s2 = set(list2)
    inters = len(s1.intersection(s2))
    if inters == 0:
        return 0
    return 2 * inters / (len(s1) + len(s2))


def anchor_distance(a, b, X, f=jaccard_similarity):
    '''
    Distance between two anchors.
    '''
    A = eval_anchor(a.data['anchor'], X)
    B = eval_anchor(b.data['anchor'], X)
    return 1 - f(A.index, B.index)


def ale_to_expl(x, e):
    '''
    Transform ALE to a local explanation for an instance.
    '''
    ales = []
    for i, fi in enumerate(e.feature_names):
        maxi = np.argmax(e.feature_values[i] > x[i])
        if x[i] - e.feature_values[i][maxi - 1] > e.feature_values[i][maxi - 1] - x[i]:
            ales.append(e.ale_values[i][maxi - 1][0])
        else:
            ales.append(e.ale_values[i][maxi][0])
    return np.array(ales)


def ale_explain_all(X, e):
    '''
    Explain all instances using ALE explanations.
    '''
    return np.array([ale_to_expl(x, e) for x in X])


def ale_distance(x, y, e):
    '''
    Distance between two ALE explanations.
    '''
    a = ale_to_expl(x, e)
    b = ale_to_expl(y, e)
    return scipy.spatial.distance.euclidean(a, b)


def robustness(X, model, snr, explain_all, metric, orig_columns, maxv=None, y_pred=None, explanations=None,
               subset_n=10, samples=10):
    '''
    Compute the robustness of a model on a dataset for an explainer.
    '''
    acc = 0
    cnt = 0
    ohe_idx = get_ohe_columns(X.columns, orig_columns)
    num_idx = [i for i in range(len(X.columns)) if i not in ohe_idx]
    std_n = snr_to_std(X.iloc[:, num_idx], snr)

    # Compute predictions and explanations if not given
    if y_pred is None:
        y_pred = model.predict(X)
    if explanations is None:
        explanations = explain_all(X.values)
    if maxv is None:
        with torch.no_grad():
            maxv = agg_exp_metric(explanations, y_pred, metric)

    # Sample perturbed data from a random subset of subset_n instances
    # for i in tqdm.tqdm_notebook(np.random.choice(X.index, size=subset_n)):
    for i in np.random.choice(len(X), size=subset_n):
        e = explanations[i]
        if len(ohe_idx) > 0:
            pert_x = sample_mixed(X.values[i], y_pred[i], model, std_n, orig_columns, X.columns, samples=samples)
        else:
            pert_x = sample(X.values[i], y_pred[i], model, std_n, samples=samples)

        if len(pert_x) > 0:  # Skip if no perturbed data survive the filter
            pert_e = explain_all(pert_x)  # Compute the explanations of the perturbed data
            dist_e = 0
            for e2 in pert_e:
                dist_e += metric(e, e2)  # Distance between perturbed and original explanation
            acc += 1 - (dist_e / len(pert_x)) / maxv  # Accumulate ratio of average expl distance and max distance
            cnt += 1

    return acc / cnt


def robustness_torch(X, model, explainer, num_idx, ohe_groups, d_E=None, maxv=None,
                     explanations=None, m=None, k=10, delta=0.2, layer=None, seed=42, fixed_mask=None, details=False):
    '''
    Compute the robustness of a model on a dataset for an explainer.

    fixed_mask: optional bool mask (m*k,) from a previous call (identical seed,
        k, delta, m and X -- the perturbations are regenerated deterministically).
        The default mask follows the CURRENT model, so a rising valid_rate
        admits larger perturbations into the average and inflates the measured
        distance even when per-perturbation stability improved; a fixed mask
        keeps the conditioning set constant for before/after comparisons.
    details: return a dict instead of the (value, valid_rate) tuple:
        robustness (own mask), robustness_fixed (fixed/common mask),
        robustness_sq (mean SQUARED distance, the CRMO regularizer's form),
        valid_rate (current model's rate), mask, per-perturbation dist.
    '''
    rs = np.random.RandomState(seed)
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    d_E = d_E if d_E is not None else _euclidean
    m = len(X) if m is None else m

    was_training = model.training
    model.eval()
    try:
        # E_theta: normalize the explainer spec to a differentiable callable
        explain_fn = make_explainer(explainer, model, layer=layer)
        n, f = X.shape

        # ---- Sample candidate perturbations ----
        idx = torch.from_numpy(rs.randint(0, len(X), size=m))
        pert = sample_perturbations(X[idx], k, num_idx, ohe_groups, noise_std=delta, g=g)
        X_pert_all = (X[idx].unsqueeze(1) - pert).reshape(-1, f)

        # ---- Select valid prediction-preserving pairs Q ----
        with torch.no_grad():
            cls_orig = _predict_classes(model(X[idx]))
            cls_pert = _predict_classes(model(X_pert_all))
        cur_mask = cls_pert == cls_orig.repeat_interleave(k)
        valid_rate = float(cur_mask.float().mean())
        if fixed_mask is not None:
            if tuple(fixed_mask.shape) != tuple(cur_mask.shape):
                raise ValueError(f"fixed_mask has shape {tuple(fixed_mask.shape)}, "
                                 f"expected {tuple(cur_mask.shape)} -- use the same "
                                 "seed, k, delta, m and X as the run that produced it.")
            fixed_mask = fixed_mask.to(cur_mask.device)
            sel = cur_mask | fixed_mask
        else:
            sel = cur_mask

        # ---- Explanations and regularizers ----
        if explanations is None:
            explanations = explain_fn(X[idx])
        E_X = _attribution_matrix(explanations[idx], m, f)

        if maxv is None:
            maxv = compute_maxv(explanations)

        if not sel.any():
            raise ValueError("No perturbed instances survive the prediction-preserving filter")

        # ---- Robustness ----
        X_pert = X_pert_all[sel]
        E_pert = _attribution_matrix(explain_fn(X_pert), len(X_pert), f, detach=False)
        sel_rows = torch.nonzero(sel, as_tuple=True)[0]
        E_orig = E_X[sel_rows // k]
        dist = 1 - d_E(E_orig, E_pert) / maxv

        own = cur_mask[sel_rows]
        fixed = fixed_mask[sel_rows] if fixed_mask is not None else own
        if details:
            return {"robustness": float(dist[own].mean()) if own.any() else float("nan"),
                    "robustness_fixed": float(dist[fixed].mean()) if fixed.any() else float("nan"),
                    # "robustness_sq": float((dist[own] ** 2).mean()) if own.any() else float("nan"),
                    "valid_rate": valid_rate, "mask": cur_mask, "dist": dist.detach()}
        if fixed_mask is not None:
            if not fixed.any():
                raise ValueError("No perturbed instances survive the fixed mask")
            return float(dist[fixed].mean()), valid_rate
        return float(dist.mean()), valid_rate

    finally:
        if was_training:
            model.train()

    # Sample perturbed data from a random subset of subset_n instances
    # for i in tqdm.tqdm_notebook(np.random.choice(X.index, size=subset_n)):
    for i in np.random.choice(len(X), size=m):
        e = explanations[i]
        if len(ohe_idx) > 0:
            pert_x = sample_mixed_torch(X[i], y_pred[i], model, std_n, orig_columns, x_columns, samples=k)
        else:
            pert_x = sample_torch(X[i], y_pred[i], model, std_n, samples=k)

        if len(pert_x) > 0:  # Skip if no perturbed data survive the filter
            pert_e = explain_all(pert_x).numpy()  # Compute the explanations of the perturbed data
            dist_e = 0
            for e2 in pert_e:
                dist_e += torch.clamp(metric(e, e2) / maxv,
                                      max=1.0)  # Distance between perturbed and original explanation
            if surrogate:
                acc += dist_e / len(pert_x)
            else:
                acc += 1 - dist_e / len(pert_x)  # Accumulate ratio of average expl distance and max distance
            cnt += 1

    return acc / cnt


def robust_shap(X, model, E, snr, orig_columns, is_twoclass, **params):
    '''
    Compute the robustness of a model on a dataset using a SHAP explainer.
    '''
    if is_twoclass:
        explain_all = lambda X, E: E.shap_values(X)[0]
    else:
        explain_all = lambda X, E: E.shap_values(X)
    metric = scipy.spatial.distance.euclidean
    return robustness(X, model, E, snr, explain_all, metric, orig_columns, **params)


def robust_lime(X, model, E, snr, orig_columns, labels=(1,), **params):
    '''
    Compute the robustness of a model on a dataset using a lime tabular explainer.
    '''
    explain_all = lambda X, E: lime_explain_all(X, model, E, labels)
    metric = scipy.spatial.distance.euclidean
    return robustness(X, model, E, snr, explain_all, metric, orig_columns, **params)


def robust_anchor(X, X_ohe, model, E, snr, orig_columns, **params):
    '''
    Compute the robustness of a model on a dataset using a Anchor tabular explainer.
    '''
    explain_all = lambda X, E: anchor_explain_all(X, E)
    metric = lambda x, y: anchor_distance(x, y, X)
    return robustness(X_ohe, model, E, snr, explain_all, metric, orig_columns, maxv=1, **params)


def robust_ale(X, model, E, snr, orig_columns, explanations=None, **params):
    '''
    Compute the robustness of a model on a dataset using an ALE tabular explainer.
    '''
    # explanations = ale explanation (ale + feature values, etc.)
    if explanations is None:
        explanations = E.explain(X.values)
    explain_all = lambda X, E: ale_explain_all(X, explanations)
    metric = scipy.spatial.distance.euclidean

    e = ale_explain_all(X.values, explanations)  # array of transformed, local explanations
    return robustness(X, model, E, snr, explain_all, metric, orig_columns,
                      explanations=e, **params)


def perturb_img(image, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def grad_cam_explain_all(X, model, E, **kwargs):
    return [E.explain((img, None), model, model.predict(img).argmax(), **kwargs) for img in X]


def grad_cam_explain_all_torch(X, model, E, **kwargs):
    for img in X:
        print(img.shape)
        break
    return [E.explain((img, None), model, model(img).argmax(axis=1), **kwargs) for img in X]


def robustness_img(X, model, E, snr, explain_all, metric, maxv=None, y_pred=None, explanations=None,
                   subset_n=10, samples=10, surrogate=True, **kwargs):
    acc = 0
    cnt = 0
    std_n = snr_to_std(X, snr)

    # Compute predictions and explanations if not given
    if y_pred is None:
        y_pred = model.predict(X).argmax(axis=1)
    if explanations is None:
        explanations = explain_all(np.expand_dims(X, axis=1), model, E, **kwargs)
    if maxv is None:
        maxv = agg_exp_metric(explanations, y_pred, metric)

    # Sample perturbed data from a random subset of subset_n instances
    # for i in tqdm.tqdm_notebook(np.random.choice(X.index, size=subset_n)):
    for i in np.random.choice(len(X), size=subset_n):
        e = explanations[i]
        pert_x = np.array([perturb_img(X[i], std=std_n) for _ in range(samples)])
        pert_y = model.predict(pert_x).argmax(axis=1)
        pert_x = pert_x[pert_y == y_pred[i]]

        if len(pert_x) > 0:  # Skip if no perturbed data survive the filter
            pert_e = explain_all(np.expand_dims(pert_x, axis=1), model, E,
                                 **kwargs)  # Compute the explanations of the perturbed data
            dist_e = 0
            for e2 in pert_e:
                dist_e += metric(e, e2)  # Distance between perturbed and original explanation
            if surrogate:
                acc += (dist_e / len(pert_x)) / maxv
            else:
                acc += 1 - (dist_e / len(pert_x)) / maxv  # Accumulate ratio of average expl distance and max distance
            cnt += 1

    return acc / cnt


def robustness_graph(dataset: ShapeGGen,
                     model: torch.nn.Module,
                     criterion: torch.nn.Module,
                     exp_method: str,
                     metric: Callable,
                     pgex: PGExplainer = None,
                     node_indices: torch.Tensor = None,
                     maxv=None,
                     y_pred=None,
                     explanations=None,
                     subset_n=10,
                     samples=10,
                     **kwargs):
    acc = 0
    cnt = 0
    data = dataset.get_graph(use_fixed_split=True)
    X = data.x.to(device)
    Y = data.y.to(device)
    EIDX = data.edge_index.to(device)

    # Compute predictions and explanations if not given
    if y_pred is None:
        y_pred = model(X, EIDX)
    if explanations is None:
        if node_indices is None:
            node_indices = torch.arange(dataset.num_nodes)
        explanations = graph_explain_all(dataset, model, criterion, X, Y, EIDX, exp_method, y_pred, node_indices,
                                         pgex=pgex)
    if maxv is None:
        maxv = agg_exp_metric(explanations, y_pred.argmax(dim=1), metric)

    # Sample perturbed data from a random subset of subset_n instances
    # for i in tqdm.tqdm_notebook(np.random.choice(X.index, size=subset_n)):
    for _ in range(subset_n):
        node_idx, gt_exp = dataset.choose_node(split='test')
        pred_class = y_pred[node_idx, :].argmax(dim=0)
        node_idx_e = (node_indices == node_idx).nonzero(as_tuple=True)[0]
        e = explanations[
            node_idx_e]  # explanations is not full dataset just node_indices many --> nodeidx does not work

        pert_e = []
        for _ in range(samples):
            pert_x = data.x.to(device).clone()
            pert_x[node_idx] = perturb_node_features(pert_x, node_idx, pert_feat=torch.arange(data.x.shape[1]),
                                                     device=device)
            pert_y = model(pert_x, EIDX)[node_idx].reshape(-1, 1).argmax(dim=0)
            if pert_y == pred_class:
                pert_e.append(graph_explain_single(dataset, model, criterion, pert_x, Y, EIDX, exp_method, y_pred,
                                                   node_idx, pgex=pgex))

        dist_e = 0
        for e2 in pert_e:
            dist_e += metric(e, e2)  # Distance between perturbed and original explanation
        if len(pert_e) == 0:
            continue
        acc += 1 - (dist_e / len(pert_e)) / maxv  # Accumulate ratio of average expl distance and max distance
        cnt += 1
    return acc / cnt

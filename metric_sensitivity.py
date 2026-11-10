import copy
import math
import warnings

import numpy as np
import pandas as pd
import torch
import tqdm
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from crmo import make_explainer, _predict_classes
from stability import compute_cvi
from tabular_metrics import _attribution_matrix
from util import match_columns

torch.set_default_dtype(torch.float64)

seed = 199
torch.manual_seed(seed)
np.random.seed(seed)

DATA_FILE = "data/classified_cohort_XGBoost_scaled_complete_avg_cohort_oversampling_smote_11_44_52.csv"

FEATURE_COLS = ['age', 'Anion Gap', 'Bicarbonate', 'Chloride (whole blood)',
                'Creatinine', 'electivesurgery', 'gcs', 'gender', 'Heart Rate',
                'mechvent', 'O2 saturation pulseoxymetry', 'oasis',
                'Sodium (whole blood)', 'stroke_type', 'White Blood Cells',
                'admission_type_0', 'admission_type_1', 'admission_type_2',
                'dbsource_-1', 'dbsource_0', 'dbsource_1', 'ethnicity_0',
                'ethnicity_1', 'ethnicity_2', 'ethnicity_3', 'ethnicity_4',
                'ethnicity_5']

ORIG_COLUMNS = ['age', 'Anion Gap', 'Bicarbonate', 'Chloride (whole blood)',
                'Creatinine', 'electivesurgery', 'gcs', 'gender', 'Heart Rate',
                'mechvent', 'O2 saturation pulseoxymetry', 'oasis',
                'Sodium (whole blood)', 'stroke_type', 'White Blood Cells',
                'admission_type', 'dbsource', 'ethnicity']

# ---------------------- experimental setup (tabular.py) ---------------------
DELTA = 0.2                 # baseline neighborhood radius
ROB_M = 64                  # sampled instances for robustness
ROB_K = 100                 # perturbations per instance
S_MIN_K, S_MAX_K = 2, 10    # k-grid of the stability metric

# Trim this list to speed the study up; rankings need >= 3 explainers.
EXPLAINERS = ["grad", "ixg", "smoothgrad", "smoothgrad_sq", "ig", "gxa", "gradshap"]

# ------------------------- sensitivity grids (OFAT) -------------------------
# (name, algorithm, input distance, explanation distance, n_inst or None=all)
STAB_CONFIGS = [
    ("baseline (kmeans, n=500)", "kmeans", None, None, 500),
    ("algo=agg-avg", "agg-average", "euclidean", "euclidean", 500),
    ("algo=agg-complete", "agg-complete", "euclidean", "euclidean", 500),
    ("d_X=cityblock", "agg-average", "cityblock", "euclidean", 500),
    ("d_X=cosine", "agg-average", "cosine", "euclidean", 500),
    ("d_E=cityblock", "agg-average", "euclidean", "cityblock", 500),
    ("d_E=cosine", "agg-average", "euclidean", "cosine", 500),
    # ("n_inst=100", "kmeans", None, None, 100),
    # ("n_inst=2000", "kmeans", None, None, 2000),
    # ("n_inst=all", "kmeans", None, None, None),
]

# (name, m instances, n pairs, input distance, explanation distance)
DISC_CONFIGS = [
    ("baseline (m=64, pairs=512, L2/L2)", 64, 512, "l2", "l2"),
    ("m=32", 32, 512, "l2", "l2"),
    ("m=128", 128, 512, "l2", "l2"),
    ("m=256", 256, 512, "l2", "l2"),
    ("pairs=128", 64, 128, "l2", "l2"),
    ("pairs=2048", 64, 2048, "l2", "l2"),
    ("d_X=L1", 64, 512, "l1", "l2"),
    ("d_X=cosine", 64, 512, "cosine", "l2"),
    ("d_E=L1", 64, 512, "l2", "l1"),
    ("d_E=cosine", 64, 512, "l2", "cosine"),
]

_ROB_BASE = dict(delta=DELTA, distribution="gaussian", maxv_quantile=0.95,
                 k=ROB_K, pair_selection="prediction_preserving")
ROB_CONFIGS = [
    ("baseline", dict(_ROB_BASE)),
    ("delta=0.1", {**_ROB_BASE, "delta": 0.1}),
    ("delta=0.4", {**_ROB_BASE, "delta": 0.4}),
    ("perturb=uniform", {**_ROB_BASE, "distribution": "uniform"}),
    ("perturb=laplace", {**_ROB_BASE, "distribution": "laplace"}),
    ("z_E=q0.50", {**_ROB_BASE, "maxv_quantile": 0.50}),
    ("z_E=q0.99", {**_ROB_BASE, "maxv_quantile": 0.99}),
    ("z_E=q1.00", {**_ROB_BASE, "maxv_quantile": 1.00}),
    ("k=25", {**_ROB_BASE, "k": 25}),
    ("k=200", {**_ROB_BASE, "k": 200}),
    ("criterion=all", {**_ROB_BASE, "pair_selection": "all"}),
]


# --------------------------------- data -------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    X_df = df[FEATURE_COLS]
    y_df = df[["class"]]

    X_train, X_val, y_train, y_val = train_test_split(
        X_df.values, y_df.values, test_size=0.2, random_state=42, stratify=y_df.values
    )
    X_all = torch.tensor(X_df.values)
    y_all = torch.tensor(y_df.values)
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)

    _, counts = torch.unique(y_all, return_counts=True)
    class_weights = [counts.sum() / c for c in counts]
    samples_weight = [class_weights[int(c[0])] for c in y_train]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
    return X_all, y_all, train_loader, val_loader


# --------------------------------- model ------------------------------------
class Classifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self(x)


def evaluate_model(model, val_loader, criterion):
    val_loss = 0.0
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
            predicted = (preds >= 0.5).float()
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    return val_loss / len(val_loader), correct / total


def pretrain(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_epoch, max_accuracy, best_state = 0, 0.0, None
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        if val_accuracy > max_accuracy:
            max_accuracy, best_epoch = val_accuracy, epoch
            best_state = copy.deepcopy(model.state_dict())
        pbar.set_description(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
            f"Max Acc: {max_accuracy:.4f} | Best Epoch: {best_epoch}"
        )
    model.load_state_dict(best_state)
    return model


# ------------------------------ shared helpers ------------------------------
def pairwise_distance_matrix(Z, d_name):
    '''Pairwise distance matrix (numpy) for precomputed-linkage clustering.'''
    Zt = torch.as_tensor(Z, dtype=torch.float64)
    if d_name == "euclidean":
        D = torch.cdist(Zt, Zt, p=2)
    elif d_name == "cityblock":
        D = torch.cdist(Zt, Zt, p=1)
    elif d_name == "cosine":
        Zn = Zt / Zt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        D = 1.0 - Zn @ Zn.T
    else:
        raise ValueError(f"unknown distance '{d_name}'")
    D = D.clamp_min(0)
    D = 0.5 * (D + D.T)  # enforce symmetry for the clustering implementations
    return D.numpy()


def fit_cluster(Z, n_clusters, algorithm, d_name):
    '''Cluster labels under the given algorithm / distance implementation.'''
    Z_np = Z.numpy() if torch.is_tensor(Z) else np.asarray(Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KMeans convergence warnings on tiny clusters
        if algorithm == "kmeans":
            return KMeans(n_clusters=n_clusters, random_state=seed).fit(Z_np).labels_
        linkage = "average" if algorithm == "agg-average" else "complete"
        D = pairwise_distance_matrix(Z_np, d_name)
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage,
                                       metric="precomputed").fit(D).labels_


def maxv_quantiles(A, quantiles):
    '''z_E normalization constants from the pairwise-distance distribution.'''
    pdm = torch.cdist(A, A, p=2.0)
    iu = torch.triu_indices(len(A), len(A), offset=1)
    vals = pdm[iu[0], iu[1]]
    return {q: float(vals.quantile(q).clamp_min(1e-12)) for q in quantiles}


def pair_distance(a, b, name):
    if name == "l2":
        return (a - b).norm(dim=-1)
    if name == "l1":
        return (a - b).abs().sum(dim=-1)
    if name == "cosine":
        an = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        bn = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return 1.0 - (an * bn).sum(dim=-1)
    raise ValueError(f"unknown distance '{name}'")


def similarity_discrepancy(X_sub, E_sub, ii, jj, d_X, d_E):
    '''
    The L_stab quantity of CRMO on detached matrices: squared discrepancy
    between the normalized input- and explanation-space distance profiles
    over a fixed set of instance pairs (identical for all explainers).
    '''
    d_in = pair_distance(X_sub[ii], X_sub[jj], d_X)
    d_ex = pair_distance(E_sub[ii], E_sub[jj], d_E)
    w_in = 1 - d_in / d_in.max().clamp_min(1e-12)
    w_ex = 1 - d_ex / d_ex.max().clamp_min(1e-12)
    return float(((w_ex - w_in) ** 2).mean())


# ---------------------------- robustness variant ----------------------------
def sample_perturbations_dist(X, n_perturb, num_idx, ohe_groups, noise_std,
                              distribution, g):
    '''
    tabular_metrics.sample_perturbations with a distribution knob for the
    numeric noise (unit-variance gaussian / uniform / laplace); the OHE
    category-switch mechanism is unchanged.
    '''
    m, f = X.shape
    delta = torch.zeros(m, n_perturb, f, dtype=X.dtype)
    if len(num_idx) > 0:
        sigma = X[:, num_idx].std(dim=0)
        sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
        shape = (m, n_perturb, len(num_idx))
        if distribution == "gaussian":
            noise = torch.randn(shape, generator=g, dtype=X.dtype)
        elif distribution == "uniform":
            noise = (torch.rand(shape, generator=g, dtype=X.dtype) - 0.5) * math.sqrt(12.0)
        elif distribution == "laplace":
            u = (torch.rand(shape, generator=g, dtype=X.dtype) - 0.5).clamp(-0.499999, 0.499999)
            noise = -torch.sign(u) * torch.log1p(-2 * u.abs()) / math.sqrt(2.0)
        else:
            raise ValueError(f"unknown perturbation distribution '{distribution}'")
        delta[:, :, num_idx] = noise * (noise_std * sigma)
    for cols in ohe_groups:
        kk = len(cols)
        orig = X[:, cols]
        orig_cat = orig.argmax(dim=1)
        r = torch.randint(0, kk, (m, n_perturb), generator=g)
        clash = r == orig_cat.unsqueeze(1)
        while clash.any():
            new = torch.randint(0, kk, clash.shape, generator=g)
            r[clash] = new[clash]
            clash = r == orig_cat.unsqueeze(1)
        onehot = F.one_hot(r, kk).to(X.dtype)
        delta[:, :, cols] = orig.unsqueeze(1) - onehot
    return delta


def robustness_local(X, model, explainer, num_idx, ohe_groups, exp_all, maxv,
                     delta=DELTA, distribution="gaussian", k=ROB_K, m=ROB_M,
                     pair_selection="prediction_preserving", seed=seed):
    '''
    Local mirror of robustness_torch with the sensitivity knobs exposed
    (perturbation distribution, acceptance criterion, radius, perturbation
    count). The baseline configuration reproduces robustness_torch's protocol.
    '''
    rs = np.random.RandomState(seed)
    g = torch.Generator().manual_seed(seed)
    explain_fn = make_explainer(explainer, model)
    _, f = X.shape

    was_training = model.training
    model.eval()
    try:
        idx = torch.from_numpy(rs.randint(0, len(X), size=m))
        pert = sample_perturbations_dist(X[idx], k, num_idx, ohe_groups,
                                         noise_std=delta, distribution=distribution, g=g)
        X_pert_all = (X[idx].unsqueeze(1) - pert).reshape(-1, f)

        with torch.no_grad():
            cls_orig = _predict_classes(model(X[idx]))
            cls_pert = _predict_classes(model(X_pert_all))
        if pair_selection == "prediction_preserving":
            q_mask = cls_pert == cls_orig.repeat_interleave(k)
        else:  # "all": keep every candidate perturbation
            q_mask = torch.ones_like(cls_pert, dtype=torch.bool)
        valid_rate = float(q_mask.float().mean())
        if not q_mask.any():
            raise ValueError("no perturbations accepted; reduce delta or relax the criterion")

        E_X = exp_all[idx]
        X_pert = X_pert_all[q_mask]
        E_pert = _attribution_matrix(explain_fn(X_pert), len(X_pert), f)
        sel_rows = torch.nonzero(q_mask, as_tuple=True)[0]
        E_orig_rows = E_X[sel_rows // k]
        dist = 1 - (E_orig_rows - E_pert).norm(dim=-1) / maxv
        return {"robustness": float(dist.mean()), "valid_rate": valid_rate}
    finally:
        if was_training:
            model.train()


# ------------------------------ rank reporting ------------------------------
def rank_summary(pivot, higher_better):
    '''Rank-stability summary of every configuration against the first column.'''
    baseline_col = pivot.columns[0]
    base = pivot[baseline_col].astype(float)
    rows = []
    for col in pivot.columns:
        col_vals = pivot[col].astype(float)
        rel = (col_vals - base).abs() / base.abs().clip(lower=1e-12)
        best = col_vals.idxmax() if higher_better else col_vals.idxmin()
        rows.append({
            "config": str(col),
            "spearman_vs_baseline": float(spearmanr(base, col_vals)[0]),
            "kendall_vs_baseline": float(kendalltau(base, col_vals)[0]),
            "mean_abs_rel_change": float(rel.mean()),
            "max_abs_rel_change": float(rel.max()),
            "best_explainer": best,
        })
    return pd.DataFrame(rows)


def pivot_by_config(df, value_col, config_order):
    pv = df.pivot(index="explainer", columns="config", values=value_col)
    return pv.reindex(columns=[c for c in config_order if c in pv.columns]).round(4)


# ------------------------------- sensitivity --------------------------------
def stability_configs_effective(n_total):
    '''Deduplicate configs whose instance count exceeds the dataset size.'''
    seen, out = set(), []
    for name, algorithm, d_X, d_E, n_inst in STAB_CONFIGS:
        n_eff = n_total if n_inst is None else min(n_inst, n_total)
        key = (algorithm, d_X, d_E, n_eff)
        if key in seen:
            continue
        seen.add(key)
        label = name if n_inst is None or n_eff == n_inst else f"{name} (n={n_eff})"
        out.append((label, algorithm, d_X, d_E, n_eff))
    return out


def run_stability_sensitivity(X_all, exp_by_explainer):
    '''
    Cluster-agreement stability under varying algorithm / distances /
    instance count. Data-side clusterings are computed once per config and
    shared across explainers.
    '''
    ks = list(range(S_MIN_K, S_MAX_K + 1))
    n_total = len(X_all)
    configs = stability_configs_effective(n_total)
    idx_pool = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed + 7))

    rows = []
    for cfg_name, algorithm, d_X, d_E, n_eff in configs:
        idx = idx_pool[:n_eff]
        X_np = X_all[idx].detach().numpy()
        A_labels = {k: fit_cluster(X_np, k, algorithm, d_X) for k in ks}
        for E in EXPLAINERS:
            E_np = exp_by_explainer[E][idx].numpy()
            B_labels = {k: fit_cluster(E_np, k, algorithm, d_E) for k in ks}
            models = {k: [A_labels[k], B_labels[k]] for k in ks}
            cvi = compute_cvi(models, S_MIN_K, S_MAX_K)
            rows.append({"explainer": E, "config": cfg_name,
                         "ARI_max": float(cvi[0].max()), "AMI_max": float(cvi[1].max()),
                         "V_max": float(cvi[2].max()), "ARI_mean": float(cvi[0].mean()),
                         "AMI_mean": float(cvi[1].mean()), "V_mean": float(cvi[2].mean())})
        print(f"  stability config done: {cfg_name}")
    return pd.DataFrame(rows), [c[0] for c in configs]


def run_discrepancy_sensitivity(X_all, exp_by_explainer):
    '''
    Structural similarity-profile discrepancy under varying instance count,
    pair count and distance functions. The instance subsample and the pairs
    are fixed across explainers.
    '''
    m_max = min(max(cfg[1] for cfg in DISC_CONFIGS), len(X_all))
    p_max = max(cfg[2] for cfg in DISC_CONFIGS)
    idx_disc = torch.randperm(len(X_all),
                              generator=torch.Generator().manual_seed(seed + 13))[:m_max]
    X_disc = X_all[idx_disc]

    pairs = {}
    for mm in sorted({min(cfg[1], len(X_all)) for cfg in DISC_CONFIGS}):
        rp = np.random.RandomState(seed + 11)
        ii = torch.from_numpy(rp.randint(0, mm, size=p_max))
        jj = torch.from_numpy(rp.randint(0, mm, size=p_max))
        keep = ii != jj
        pairs[mm] = (ii[keep], jj[keep])

    rows = []
    for cfg_name, mm, n_pairs, d_X, d_E in DISC_CONFIGS:
        mm_eff = min(mm, len(X_all))
        ii, jj = (t[:n_pairs] for t in pairs[mm_eff])
        X_sub = X_disc[:mm_eff]
        for E in EXPLAINERS:
            E_sub = exp_by_explainer[E][idx_disc][:mm_eff]
            rows.append({"explainer": E, "config": cfg_name,
                         "SimDiscrepancy": similarity_discrepancy(X_sub, E_sub, ii, jj, d_X, d_E)})
    return pd.DataFrame(rows), [c[0] for c in DISC_CONFIGS]


def run_robustness_sensitivity(X_all, model, exp_by_explainer, num_idx, ohe_groups):
    '''Robustness under varying radius / distribution / z_E / k / criterion.'''
    quantiles = sorted({cfg["maxv_quantile"] for _, cfg in ROB_CONFIGS})
    maxv_by_E = {E: maxv_quantiles(exp_by_explainer[E], quantiles) for E in EXPLAINERS}

    rows, valid_rows = [], []
    for cfg_name, cfg in ROB_CONFIGS:
        for E in EXPLAINERS:
            res = robustness_local(X_all, model, E, num_idx, ohe_groups,
                                   exp_by_explainer[E], maxv_by_E[E][cfg["maxv_quantile"]],
                                   delta=cfg["delta"], distribution=cfg["distribution"],
                                   k=cfg["k"], m=ROB_M,
                                   pair_selection=cfg["pair_selection"], seed=seed)
            rows.append({"explainer": E, "config": cfg_name, "Robustness": res["robustness"]})
            valid_rows.append({"explainer": E, "config": cfg_name,
                               "valid_rate": res["valid_rate"]})
        print(f"  robustness config done: {cfg_name}")
    return pd.DataFrame(rows), pd.DataFrame(valid_rows), [c[0] for c in ROB_CONFIGS]


X_all, y_all, train_loader, val_loader = load_data()
ohe_groups = [v for v in match_columns(FEATURE_COLS, ORIG_COLUMNS).values()
              if len(v) > 1]
num_idx = [i for i in range(X_all.shape[1]) if all(i not in grp for grp in ohe_groups)]

model = Classifier(X_all.shape[1])
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-3)
model = pretrain(model, train_loader, val_loader, criterion, optimizer, epochs=100)
model.eval()

print("\nComputing explanations for all explainers on the full data ...")
n, f = X_all.shape
exp_by_explainer = {}
for E in EXPLAINERS:
    torch.manual_seed(seed)  # deterministic stochastic explainers (smoothgrad, gradshap)
    exp_by_explainer[E] = _attribution_matrix(make_explainer(E, model)(X_all), n, f)
    print(f"  {E}: done")

print("\n=== stability sensitivity (cluster agreement) ===")
stab_df, stab_order = run_stability_sensitivity(X_all, exp_by_explainer)
print("\n=== structural discrepancy sensitivity ===")
disc_df, disc_order = run_discrepancy_sensitivity(X_all, exp_by_explainer)
print("\n=== robustness sensitivity ===")
rob_df, rob_valid_df, rob_order = run_robustness_sensitivity(
    X_all, model, exp_by_explainer, num_idx, ohe_groups)

# ---- pivot tables: explainer x configuration ----
stab_pivot = pivot_by_config(stab_df, "ARI_max", stab_order)
base_cfg = stab_order[0]
cvi_table = (stab_df[stab_df["config"] == base_cfg]
             .set_index("explainer")[["ARI_max", "AMI_max", "V_max", "ARI_mean", "AMI_mean", "V_mean"]]
             .round(4))
disc_pivot = pivot_by_config(disc_df, "SimDiscrepancy", disc_order)
rob_pivot = pivot_by_config(rob_df, "Robustness", rob_order)
rob_valid_pivot = pivot_by_config(rob_valid_df, "valid_rate", rob_order)

print("\n=== Stability (ARI, max over k-grid) per configuration ===")
print(stab_pivot.to_string())
print("\n=== Stability: cluster validity index choice (baseline clustering) ===")
print(cvi_table.to_string())
print("\n=== Structural discrepancy per configuration (lower = better) ===")
print(disc_pivot.to_string())
print("\n=== Robustness per configuration ===")
print(rob_pivot.to_string())
print("\n=== Robustness: accepted-neighbor rate per configuration ===")
print(rob_valid_pivot.to_string())

# ---- rank-stability summary ----
summary_df = pd.concat([
    rank_summary(stab_pivot, higher_better=True).assign(
        family="stability (ARI, max over k-grid)"),
    rank_summary(cvi_table, higher_better=True).assign(
        family="stability: CVI choice (same clustering)"),
    rank_summary(disc_pivot, higher_better=False).assign(
        family="structural discrepancy (lower = better)"),
    rank_summary(rob_pivot, higher_better=True).assign(
        family="robustness"),
], ignore_index=True).round(4)

print("\n=== Explainer-ranking stability across implementation choices ===")
print(summary_df.to_string(index=False))
print("\nInterpretation: spearman/kendall close to 1.0 -> the explainer ranking")
print("is insensitive to the implementation choice; large relative score changes")
print("with stable rankings -> the absolute numbers depend on the choice, the")
print("comparison does not. (z_E and delta changes shift scores by construction.)")

stab_pivot.to_csv("results/sensitivity_stability.csv")
cvi_table.to_csv("results/sensitivity_stability_cvi.csv")
disc_pivot.to_csv("results/sensitivity_discrepancy.csv")
rob_pivot.to_csv("results/sensitivity_robustness.csv")
rob_valid_pivot.to_csv("results/sensitivity_robustness_validrate.csv")
summary_df.to_csv("results/sensitivity_summary.csv", index=False)
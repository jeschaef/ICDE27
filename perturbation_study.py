import copy
import warnings

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from crmo import make_explainer, compute_maxv, _euclidean, explainer_name
from robustness import robustness_torch
from stability import stability
from tabular_metrics import (sample_perturbations, infidelity, max_sensitivity,
                             faithfulness_correlation)
from util import match_columns

# ------------------------------ configuration ------------------------------
seed = 199
EXPLAINER = "gradshap"
noise_small = 0.05        # small noise: fraction of the per-feature explanation std
noise_large = 0.50        # large noise: fraction of the per-feature explanation std

delta = 0.2                 # perturbation radius used by the metrics
m = 64                      # number of sampled instances (robustness)
k = 100                     # perturbations per instance
subset_n = 100              # evaluation subset size
n_subsets = 50              # random removal subsets per instance (faithfulness)

crmo_lr = 1e-3              # CRMO learning rate (use same optimizer for momentum)

num_epochs = 100
pretrain_lr = 1e-3
batch_size = 64

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

COL_ORDER = ["variant", "rel_dist_to_orig", "ValAcc", "Stability", "Robustness",
             "valid_rate", "Infidelity", "MaxSensitivity", "Faithfulness"]

VARIANT_DOCS = {
    "original": "unmodified explanation E",
    "small_noise": f"E + N(0, ({noise_small}*std_E)^2)",
    "large_noise": f"E + N(0, ({noise_large}*std_E)^2)",
    "instance_permutation": "explanations exchanged between instances (E[perm])",
    "constant": "E[0] repeated for every instance",
    "random": "unrelated N(0, std_E^2) explanations (scale-matched)",
}


# --------------------------------- data ------------------------------------
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
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    return X_all, y_all, train_loader, val_loader


# --------------------------------- model -----------------------------------
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


# ------------------------- controlled variants ------------------------------
def make_variants(explanations, small_scale=noise_small, large_scale=noise_large, seed=seed):
    '''
    Build the controlled explanation variants. Noise is scaled per feature by
    std_E so the added perturbation size is comparable across attribution
    magnitudes; the random variant is matched to the same scale so it differs
    from the original only in content, not magnitude.
    '''
    n, f = explanations.shape
    g = torch.Generator().manual_seed(seed)
    sigma = explanations.std(dim=0)
    sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
    return {
        "original": explanations.clone(),
        "small_noise": explanations + torch.randn(n, f, generator=g) * (small_scale * sigma),
        "large_noise": explanations + torch.randn(n, f, generator=g) * (large_scale * sigma),
        "instance_permutation": explanations[torch.randperm(n, generator=g)],
        "constant": explanations[0:1].repeat(n, 1),
        "random": torch.randn(n, f, generator=g) * sigma,
    }


# ------------------------------ evaluation ----------------------------------
def evaluate_variant(model, fn, X, E, E_orig, name, num_idx, ohe_groups, maxv, val_acc):
    # Fresh, identical RNG per variant: same subset and same perturbations
    # for every variant, so rows differ only through the explanation matrix.
    rs = np.random.RandomState(seed)
    g = torch.Generator().manual_seed(seed)

    idx = rs.choice(len(X), size=min(subset_n, len(X)), replace=False)
    X_sub = X[idx]
    E_sub = E[idx]
    pert = sample_perturbations(X_sub, k, num_idx, ohe_groups, noise_std=delta, g=g)

    # Sanity check: how far the variant moved from the original explanations.
    rel_dist = ((E - E_orig).norm(dim=-1)
                / E_orig.norm(dim=-1).clamp_min(1e-12)).mean().item()

    with warnings.catch_warnings():
        # KMeans on the constant variant warns about fewer distinct clusters
        # than requested; that degenerate clustering IS the expected signal.
        warnings.simplefilter("ignore")
        S = stability(X, E, algorithm=KMeans, min_k=2, max_k=10, seed=seed).max()

    # maxv is fixed to the ORIGINAL explanation scale so robustness changes
    # come from the variant, not from a shifting normalization constant.
    R = robustness_torch(X, model, EXPLAINER, num_idx, ohe_groups, maxv=maxv,
                         d_E=_euclidean, explanations=E, m=m, k=k, delta=delta,
                         seed=seed, details=True)

    return {
        "variant": name,
        "rel_dist_to_orig": rel_dist,
        "ValAcc": val_acc,
        "Stability": float(S),
        "Robustness": R["robustness"],
        "valid_rate": R["valid_rate"],
        "Infidelity": infidelity(X_sub, E_sub, model, pert),
        "MaxSensitivity": max_sensitivity(X_sub, E_sub, fn, pert),
        "Faithfulness": faithfulness_correlation(X_sub, E_sub, model, num_idx,
                                                 ohe_groups, n_subsets=n_subsets,
                                                 rs=rs, g=g),
    }


def check_noise_response(report_df):
    '''
    Informational check: do the metrics respond monotonically to the noise
    level (original -> small -> large)? Not an assertion -- recorded in the
    output for the study report.
    '''
    values = report_df.set_index("variant")
    checks = []
    for metric, higher_is_worse in [("Robustness", True), ("Faithfulness", True),
                                    ("Infidelity", False), ("MaxSensitivity", False)]:
        o, s, l = (values.loc[v, metric] for v in
                   ("original", "small_noise", "large_noise"))
        sign = -1.0 if higher_is_worse else 1.0  # degrade => metric moves this way
        ok = sign * (s - o) >= 0 and sign * (l - s) >= 0
        checks.append({"metric": metric, "original": o, "small_noise": s,
                       "large_noise": l, "monotone_noise_response": bool(ok)})
    return pd.DataFrame(checks)



torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)

print("Controlled explanation variants:")
for vname, desc in VARIANT_DOCS.items():
    print(f"  {vname:22s} {desc}")
print()

X_all, y_all, train_loader, val_loader = load_data()
ohe_groups = [v for v in match_columns(FEATURE_COLS, ORIG_COLUMNS).values()
              if len(v) > 1]
num_idx = [i for i in range(X_all.shape[1]) if all(i not in grp for grp in ohe_groups)]

model = Classifier(X_all.shape[1])
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=pretrain_lr)
model = pretrain(model, train_loader, val_loader, criterion, optimizer, num_epochs)

with torch.no_grad():
    _, val_acc = evaluate_model(model, val_loader, criterion)
print(f"\nPretrained model -- val accuracy: {val_acc:.4f} (fixed for all variants)\n")

fn = make_explainer(EXPLAINER, model)
with torch.no_grad():
    E_orig = fn(X_all).detach()
maxv = compute_maxv(E_orig)

variants = make_variants(E_orig.detach())
rows = []
for name, E in variants.items():
    row = evaluate_variant(model, fn, X_all.detach(), E, E_orig, name,
                           num_idx, ohe_groups, maxv, val_acc)
    rows.append(row)
    print(f"[{name:22s}] rel_dist={row['rel_dist_to_orig']:.4f} "
          f"stab={row['Stability']:.4f} rob={row['Robustness']:.4f} "
          f"infid={row['Infidelity']:.4f} sens={row['MaxSensitivity']:.4f} "
          f"faith={row['Faithfulness']:.4f}")

report_df = pd.DataFrame(rows)[COL_ORDER].round(4)
print("\n=== Perturbation study: metrics per explanation variant ===")
print(report_df.to_string(index=False))
report_df.to_csv(f"results/perturbation_study_{explainer_name(EXPLAINER)}.csv", index=False)

value_cols = [c for c in COL_ORDER if c != "variant"]
delta_df = (report_df.set_index("variant")[value_cols]
            - report_df.set_index("variant").loc["original", value_cols])
delta_df = delta_df.drop(index="original").round(4)
print("\n=== Deltas vs original (variant - original) ===")
print(delta_df.to_string())
delta_df.to_csv(f"results/perturbation_study_deltas_{explainer_name(EXPLAINER)}.csv")

check_df = check_noise_response(report_df).round(4)
print("\n=== Monotone response to noise level (informational) ===")
print(check_df.to_string(index=False))

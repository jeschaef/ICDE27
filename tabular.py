import copy

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from crmo import CRMO, make_explainer, _euclidean, compute_maxv, _predict_classes, explainer_name
from robustness import robustness_torch
from stability import stability
from tabular_metrics import evaluate_exp_quality
from util import match_columns

# print("CUDA?", torch.cuda.is_available(), torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

seed = 199
torch.manual_seed(seed)
np.random.seed(seed)

# columns: 'age', 'Anion Gap', 'Bicarbonate', 'Chloride (whole blood)',
#        'Creatinine', 'electivesurgery', 'gcs', 'gender', 'Heart Rate',
#        'mechvent', 'O2 saturation pulseoxymetry', 'oasis',
#        'Sodium (whole blood)', 'stroke_type', 'White Blood Cells',
#        'admission_type_0', 'admission_type_1', 'admission_type_2',
#        'dbsource_-1', 'dbsource_0', 'dbsource_1', 'ethnicity_0', 'ethnicity_1',
#        'ethnicity_2', 'ethnicity_3', 'ethnicity_4', 'ethnicity_5', 'class',
#        'out'
df = pd.read_csv("data/classified_cohort_XGBoost_scaled_complete_avg_cohort_oversampling_smote_11_44_52.csv")

orig_columns = ['age', 'Anion Gap', 'Bicarbonate', 'Chloride (whole blood)',
                'Creatinine', 'electivesurgery', 'gcs', 'gender', 'Heart Rate',
                'mechvent', 'O2 saturation pulseoxymetry', 'oasis',
                'Sodium (whole blood)', 'stroke_type', 'White Blood Cells',
                'admission_type', 'dbsource', 'ethnicity', ]

X_df = df[['age', 'Anion Gap', 'Bicarbonate', 'Chloride (whole blood)',
           'Creatinine', 'electivesurgery', 'gcs', 'gender', 'Heart Rate',
           'mechvent', 'O2 saturation pulseoxymetry', 'oasis',
           'Sodium (whole blood)', 'stroke_type', 'White Blood Cells',
           'admission_type_0', 'admission_type_1', 'admission_type_2',
           'dbsource_-1', 'dbsource_0', 'dbsource_1', 'ethnicity_0', 'ethnicity_1',
           'ethnicity_2', 'ethnicity_3', 'ethnicity_4', 'ethnicity_5'
           ]]
y_df = df[['class']]

X_all = torch.tensor(X_df.values)
y_all = torch.tensor(y_df.values)

X_train, X_val, y_train, y_val = train_test_split(
    X_df.values, y_df.values, test_size=0.2, random_state=42, stratify=y_df.values
)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

num_classes = 2
_, nr_samples_per_class = torch.unique(y_all, return_counts=True)
class_weights = [sum(nr_samples_per_class) / c for c in nr_samples_per_class]
samples_weight = [class_weights[int(c[0])] for c in y_train]
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, sampler=sampler)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# samples_weight2 = [class_weights[int(c[0])] for c in y_all]
# sampler2 = WeightedRandomSampler(samples_weight2, len(samples_weight2))
# loader = DataLoader(TensorDataset(X_all, y_all), batch_size=64, sampler=sampler2)


# model
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


def pretrain():
    best_epoch = 0
    max_accuracy = 0.0
    pbar = tqdm.tqdm(range(num_epochs))
    best_model = None
    for epoch in pbar:
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss, val_accuracy = evaluate_model()
        if max_accuracy < val_accuracy:
            max_accuracy = val_accuracy
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())

        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Max Acc: {max_accuracy:.4f} | "
            f"Best Epoch: {best_epoch}"
        )
    model.load_state_dict(best_model)
    return model


def evaluate_model():
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
    return val_loss, correct / total


# Pretrain
num_epochs = 100
pretrain_lr = 1e-3
model = Classifier(X_all.shape[1])  # .to(device)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=pretrain_lr)

model = pretrain()


# Experimental setup
delta = 0.2                 # perturbation radius used by the metrics
m = 64                      # number of sampled instances (robustness)
k = 100                     # perturbations per instance
subset_n = 100              # evaluation subset size
n_subsets = 50              # random removal subsets per instance (faithfulness)

crmo_lr = 1e-3              # CRMO learning rate (use same optimizer for momentum)

s_algorithm=KMeans          # Stability parameters
s_min_k=2
s_max_k=10

explainers = ["grad", "ixg", "smoothgrad", "smoothgrad_sq", "ig", "gxa", "gradshap"]
lambda_configurations = [
    (0, 0), (1, 0), (0, 1), (1, 1), (3, 1), (1, 3)
]


# Feature units for the quality metrics: numeric/binary columns + OHE groups
ohe_groups = [v for v in match_columns(X_df.columns, orig_columns).values() if len(v) > 1]
num_idx = [i for i in range(X_all.shape[1]) if all(i not in g for g in ohe_groups)]
maxv_X = compute_maxv(X_all)

for E in explainers:
    d_E = _euclidean
    for lambda_s, lambda_r in lambda_configurations:
        report_rows = []

        exp_all = make_explainer(E, model)(X_all).detach()
        y_pred = _predict_classes(model(X_all))
        maxv = compute_maxv(exp_all)
        # print(maxvX, maxv)

        # Continuity before
        with torch.no_grad():
            _, val_acc = evaluate_model()
            R_b = robustness_torch(X_all, model, E, num_idx, ohe_groups, maxv=maxv, d_E=d_E, explanations=exp_all,
                                   m=m, k=k, delta=delta, seed=seed, details=True)
            S = stability(X_all, exp_all, algorithm=s_algorithm, min_k=s_min_k, max_k=s_max_k, seed=seed).max()
            metrics_before = evaluate_exp_quality(X_all, model, E, num_idx, ohe_groups,
                                                  noise_std=delta, subset_n=subset_n, n_perturb=k,
                                                  n_subsets=n_subsets, seed=seed)
            report_rows.append({"stage": "before", "Val Acc": val_acc,
                                "Stability": float(S),
                                "Robustness": R_b["robustness"],
                                "valid_rate": R_b["valid_rate"],
                                "maxv": maxv,
                                **metrics_before})

        # CRMO
        optimizer.param_groups[0]['lr'] = crmo_lr
        model, crmo_res = CRMO(model, X_train, y_train, E, criterion,
                               num_idx, ohe_groups,
                               delta=delta, lambda_s=lambda_s, lambda_r=lambda_r,
                               m=m, k=k, epochs=epochs, batch_size=m, seed=seed,
                               optimizer=optimizer,
                               log_every=1)
        # print("crmo:", {k: v for k, v in crmo_res.items() if k != "history"})

        # ---- Loss curves over the CRMO finetuning (joint + individual losses) ----
        history_df = pd.DataFrame(crmo_res["history"])
        # history_df.to_csv("results/crmo_history.csv", index=False)

        fig, (ax_j, ax_ind) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        ax_j.plot(history_df["step"], history_df["J"], color="black")
        ax_j.set_ylabel("J (joint loss)")
        ax_j.set_title(f"CRMO finetuning ({crmo_res['explainer']}, "
                       f"$\\lambda_s$={crmo_res['lambda_s']}, $\\lambda_r$={crmo_res['lambda_r']})")

        ax_ind.plot(history_df["step"], history_df["task"], label="task")
        ax_ind.plot(history_df["step"], history_df["stab"], label="stab")
        ax_ind.plot(history_df["step"], history_df["rob"], label="rob")
        ax_ind.set_xlabel("step")
        ax_ind.set_ylabel("individual losses")
        ax_ind.legend()

        fig.tight_layout()
        fig.savefig(f"results/crmo_losses_{explainer_name(E)}_lambdas_{lambda_s:.1f}_lambdar_{lambda_r:.1f}.png",
                    dpi=150)
        plt.show()

        # Continuity after
        with torch.no_grad():
            _, val_acc = evaluate_model()
            exp_all = make_explainer(E, model)(X_all).detach()
            maxv_after = compute_maxv(exp_all)
            R_a = robustness_torch(X_all, model, E, num_idx, ohe_groups, maxv=maxv, d_E=d_E, explanations=exp_all,
                                   m=m, k=k, delta=delta, seed=seed, details=True)
            S = stability(X_all, exp_all, seed=seed).max()
            metrics_after = evaluate_exp_quality(X_all, model, E, num_idx, ohe_groups,
                                                 noise_std=delta, subset_n=subset_n, n_perturb=k,
                                                 n_subsets=n_subsets, seed=seed)
            report_rows.append({"stage": "after", "Val Acc": val_acc,
                                "maxv": maxv_after,
                                "Robustness": R_a["robustness"],
                                "valid_rate": R_a["valid_rate"],
                                "Stability": float(S), **metrics_after})
            report_df = pd.DataFrame(report_rows)

        # ---- Difference row: values after minus values before finetuning ----
        diff_row = {"stage": "diff (after - before)"}
        for col in report_df.columns:
            if col == "stage":
                continue
            before = report_df.loc[report_df["stage"] == "before", col]
            after = report_df.loc[report_df["stage"] == "after", col]
            try:
                diff_row[col] = float(after.iloc[0]) - float(before.iloc[0])
            except (TypeError, ValueError, IndexError):
                diff_row[col] = None  # non-numeric columns have no difference
        report_df = pd.concat([report_df, pd.DataFrame([diff_row])], ignore_index=True)

        print(report_df.to_string(index=False))

        report_df.to_csv(f"results/{explainer_name(E)}_lambdas_{lambda_s:.1f}_lambdar_{lambda_r:.1f}.csv", index=False)
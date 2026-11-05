'''
Scalability & computational-efficiency analysis of CRMO.

Goal
----
Show how the framework behaves as the dataset grows. The cohort dataset is
resampled to the target sizes (500 / 5000 / 10000); sizes beyond the available
rows are bootstrap-resampled from the real cohort (or read from
LARGE_DATA_FILE when provided), so only n grows while the feature distribution
stays the real one. For every size a fresh classifier is pretrained and:

    * explanation generation is timed on the full dataset,
    * the stability metric (KMeans over the k-grid) is timed,
    * the robustness metric (m instances x k perturbations, n-independent
      budget) is timed,
    * CRMO finetuning is timed and reduced to time per epoch, with the
      task-only pretraining epoch as reference (overhead factor),
    * peak memory (process RSS) is sampled during each phase,
    * work counters are reported: pairwise distances behind maxv (O(n^2),
      capped), sampled instance pairs for L_stab, and candidate / valid
      perturbations for L_robu and for the robustness metric.

The compute budget (explainer, m, k, batch_size, delta, CRMO epochs) is fixed
across sizes, so time differences come from n only.

Outputs:
    results/crmo_scalability.csv        one row per dataset size
    results/crmo_scalability_times.png  phase times vs n (log-log)
'''
import copy
import gc
import threading
import time
import warnings

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

from crmo import CRMO, make_explainer, compute_maxv, _euclidean
from robustness import robustness_torch
from stability import stability
from util import match_columns

# ------------------------------ configuration ------------------------------
seed = 199
EXPLAINER = "grad"          # per-step CRMO cost scales with the explainer;
# "grad" is the cheapest differentiable surrogate
SIZES = [500, 5000, 10000]

delta = 0.2                 # perturbation radius (metrics + CRMO)
m = 64                      # sampled pairs per CRMO step / robustness instances
k = 100                     # perturbations per instance
batch_size = 64
crmo_epochs = 2             # per-epoch time = total / epochs actually run
pretrain_epochs = 25        # not a measured quantity; baseline epoch time only

lambda_s = 1.0
lambda_r = 1.0
crmo_lr = 1e-3
pretrain_lr = 1e-3

maxv_cap = 5000             # cap the O(n^2) cdist in compute_maxv on large data

DATA_FILE = "data/classified_cohort_XGBoost_scaled_complete_avg_cohort_oversampling_smote_11_44_52.csv"
LARGE_DATA_FILE = None      # optional CSV (FEATURE_COLS + 'class') for sizes
# beyond the cohort; bootstrap is used otherwise

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

COL_ORDER = ["n", "source", "val_acc",
             "explain_time_s", "maxv_time_s", "stability_time_s", "robustness_time_s",
             "pretrain_epoch_time_s", "crmo_epoch_time_s", "crmo_total_time_s",
             "crmo_epochs_run", "crmo_steps", "crmo_overhead_x",
             "maxv_pairwise_distances", "crmo_pair_comparisons",
             "crmo_candidate_perturbations", "crmo_valid_perturbations",
             "robustness_comparisons", "robustness_valid_perturbations",
             "peak_rss_mb_explain", "peak_rss_mb_stability", "peak_rss_mb_robustness",
             "peak_rss_mb_crmo"]


# --------------------------- memory measurement ----------------------------
def _make_rss_reader():
    '''
    Cross-platform current-RSS reader (MB): psutil, then the Windows PSAPI,
    then resource (Unix). ru_maxrss on macOS is reported in bytes, which only
    makes the fallback an upper bound.
    '''
    try:
        import psutil
        proc = psutil.Process()
        return lambda: proc.memory_info().rss / (1024 ** 2)
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        psapi = ctypes.windll.psapi
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        def read_windows():
            psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
            return pmc.WorkingSetSize / (1024 ** 2)
        return read_windows
    except Exception:
        pass
    import resource
    return lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


_rss_reader = _make_rss_reader()


def time_phase(fn, *args, **kwargs):
    '''
    Run fn and return (result, seconds, peak RSS in MB during the call).
    A daemon thread samples the RSS every 50 ms (numpy/torch C calls release
    the GIL, so the sampler keeps up with the heavy phases).
    '''
    peak = [_rss_reader()]
    stop = threading.Event()

    def sampler():
        while not stop.wait(0.05):
            rss = _rss_reader()
            if rss > peak[0]:
                peak[0] = rss

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    seconds = time.perf_counter() - t0
    stop.set()
    th.join()
    return result, seconds, peak[0]


# --------------------------------- data ------------------------------------
def load_cohort():
    df = pd.read_csv(DATA_FILE)
    return df[FEATURE_COLS + ["class"]]


def build_dataset(base_df, n_target):
    '''
    Subsample the cohort to n_target rows; when n_target exceeds the available
    rows, bootstrap-resample with replacement (or sample LARGE_DATA_FILE when
    provided). Only n grows; the feature distribution stays the real one.
    '''
    pool, source = base_df, "cohort"
    if n_target > len(pool) and LARGE_DATA_FILE is not None:
        pool = pd.read_csv(LARGE_DATA_FILE)[FEATURE_COLS + ["class"]]
        source = "large_file"
    replace = n_target > len(pool)
    if replace:
        source = "bootstrap"
    sample = pool.sample(n=n_target, replace=replace, random_state=seed)
    X = torch.tensor(sample[FEATURE_COLS].values, dtype=torch.float64)
    y = torch.tensor(sample[["class"]].values, dtype=torch.float64)
    return X, y, source


def make_loaders(X, y):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X.numpy(), y.numpy(), test_size=0.2, random_state=42, stratify=y.numpy().ravel())
    X_tr, y_tr = torch.as_tensor(X_tr), torch.as_tensor(y_tr)
    X_va, y_va = torch.as_tensor(X_va), torch.as_tensor(y_va)

    _, counts = torch.unique(y, return_counts=True)
    class_weights = [counts.sum() / c for c in counts]
    samples_weight = [class_weights[int(c[0])] for c in y_tr]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size)
    return train_loader, val_loader


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


def evaluate_model(model, val_loader, criterion):
    val_loss = 0.0
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
            correct += ((preds >= 0.5).float() == yb).sum().item()
            total += yb.size(0)
    return val_loss / len(val_loader), correct / total


def pretrain(model, train_loader, val_loader, criterion, optimizer, epochs):
    '''
    Task-only pretraining; a vehicle for a sensible model, not a measured
    quantity. Returns (model, mean epoch seconds), skipping the first
    (warm-up) epoch so the reference epoch time is fair.
    '''
    best_state, max_accuracy = None, 0.0
    epoch_times = []
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        t0 = time.perf_counter()
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        epoch_times.append(time.perf_counter() - t0)

        _, val_accuracy = evaluate_model(model, val_loader, criterion)
        if val_accuracy > max_accuracy:
            max_accuracy, best_state = val_accuracy, copy.deepcopy(model.state_dict())
        pbar.set_description(f"pretrain {epoch + 1}/{epochs} | val acc {val_accuracy:.4f}")
    model.load_state_dict(best_state)
    return model, float(np.mean(epoch_times[1:]))


# ------------------------------ per-size run -------------------------------
def run_size(n, base_df, ohe_groups, num_idx):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, source = build_dataset(base_df, n)
    train_loader, val_loader = make_loaders(X, y)

    model = Classifier(X.shape[1])
    criterion = nn.BCELoss()
    model, pretrain_epoch_s = pretrain(model, train_loader, val_loader, criterion,
                                       Adam(model.parameters(), lr=pretrain_lr),
                                       pretrain_epochs)
    with torch.no_grad():
        _, val_acc = evaluate_model(model, val_loader, criterion)

    row = {"n": n, "source": source, "val_acc": round(val_acc, 4)}

    # ---- explanation generation on the full dataset ----
    fn = make_explainer(EXPLAINER, model)
    E, explain_s, peak_explain = time_phase(lambda: fn(X).detach())
    row["explain_time_s"] = round(explain_s, 3)
    row["peak_rss_mb_explain"] = round(peak_explain, 1)

    # ---- maxv normalization: O(n^2) cdist, capped on large data ----
    idx_cap = torch.randperm(len(E), generator=torch.Generator().manual_seed(seed))[:min(maxv_cap, len(E))]
    maxv, maxv_s, _ = time_phase(compute_maxv, E[idx_cap])
    row["maxv_time_s"] = round(maxv_s, 3)
    row["maxv_pairwise_distances"] = len(idx_cap) * (len(idx_cap) - 1) // 2

    # ---- stability: KMeans over the k-grid on data + explanations ----
    with warnings.catch_warnings():
        # bootstrap duplicates can make KMeans warn about degenerate clusterings
        warnings.simplefilter("ignore")
        S, stab_s, peak_stab = time_phase(stability, X, E, algorithm=KMeans,
                                          min_k=2, max_k=10, seed=seed)
    row["stability_time_s"] = round(stab_s, 3)
    row["peak_rss_mb_stability"] = round(peak_stab, 1)

    # ---- robustness: m instances x k perturbations (n-independent budget) ----
    R, rob_s, peak_rob = time_phase(
        robustness_torch, X, model, EXPLAINER, num_idx, ohe_groups,
        maxv=maxv, d_E=_euclidean, explanations=E, m=m, k=k, delta=delta,
        seed=seed, details=True)
    row["robustness_time_s"] = round(rob_s, 3)
    row["peak_rss_mb_robustness"] = round(peak_rob, 1)
    row["robustness_comparisons"] = m * k
    row["robustness_valid_perturbations"] = int(round(R["valid_rate"] * m * k))

    # ---- CRMO finetuning on a frozen copy of the pretrained model ----
    model_ft = copy.deepcopy(model)
    optimizer = Adam(model_ft.parameters(), lr=crmo_lr)
    crmo_out, crmo_s, peak_crmo = time_phase(
        CRMO, model_ft, X, y, EXPLAINER, criterion, num_idx, ohe_groups,
        delta=delta, lambda_s=lambda_s, lambda_r=lambda_r,
        m=m, k=k, batch_size=batch_size, epochs=crmo_epochs,
        optimizer=optimizer, seed=seed, log_every=1000)
    _, res = crmo_out

    history = pd.DataFrame(res["history"])
    epochs_run = int(history["epoch"].max()) + 1
    steps = int(res["steps"])
    # Per epoch, the batch sizes sum to n, so candidates = epochs_run * n * k.
    candidates = epochs_run * n * k

    row["crmo_total_time_s"] = round(crmo_s, 3)
    row["crmo_epochs_run"] = epochs_run
    row["crmo_epoch_time_s"] = round(crmo_s / epochs_run, 3)
    row["crmo_steps"] = steps
    row["crmo_overhead_x"] = round(crmo_s / epochs_run / max(pretrain_epoch_s, 1e-9), 2)
    row["crmo_pair_comparisons"] = steps * m          # sampled pairs per step (L_stab)
    row["crmo_candidate_perturbations"] = candidates  # b*k per instance (L_robu)
    row["crmo_valid_perturbations"] = int(round(history["valid_rate"].mean() * candidates))
    row["peak_rss_mb_crmo"] = round(peak_crmo, 1)
    row["pretrain_epoch_time_s"] = round(pretrain_epoch_s, 3)

    print(f"  val acc={val_acc:.4f} | explain={explain_s:.2f}s stab={stab_s:.2f}s "
          f"rob={rob_s:.2f}s | crmo epoch={crmo_s / epochs_run:.2f}s "
          f"(x{row['crmo_overhead_x']} task-only) | peak RSS={peak_crmo:.0f} MB")
    return row


# ---------------------------------- main -----------------------------------
torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)

print(f"CRMO scalability: explainer={EXPLAINER}, sizes={SIZES}, m={m}, k={k}, "
      f"batch={batch_size}, delta={delta}, crmo epochs={crmo_epochs}")
print("Budget is fixed across sizes; time differences come from n only.\n")

base_df = load_cohort()
ohe_groups = [v for v in match_columns(FEATURE_COLS, ORIG_COLUMNS).values()
              if len(v) > 1]
num_idx = [i for i in range(len(FEATURE_COLS)) if all(i not in grp for grp in ohe_groups)]

rows = []
for n in SIZES:
    print(f"=== dataset size n={n} ===")
    rows.append(run_size(n, base_df, ohe_groups, num_idx))
    gc.collect()

report_df = pd.DataFrame(rows)[COL_ORDER]
print("\n=== CRMO scalability per dataset size ===")
print(report_df.to_string(index=False))
report_df.to_csv("results/crmo_scalability.csv", index=False)

print("\n=== growth between consecutive sizes (time ratio) ===")
for i in range(1, len(report_df)):
    prev, cur = report_df.iloc[i - 1], report_df.iloc[i]
    print(f"  n {int(prev['n'])} -> {int(cur['n'])} (x{cur['n'] / prev['n']:g}):")
    for col in ["explain_time_s", "maxv_time_s", "stability_time_s",
                "robustness_time_s", "crmo_epoch_time_s"]:
        print(f"    {col:22s} x{cur[col] / max(prev[col], 1e-9):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
for col, label in [("explain_time_s", "explanation generation"),
                   ("maxv_time_s", "maxv (cdist, capped)"),
                   ("stability_time_s", "stability"),
                   ("robustness_time_s", "robustness"),
                   ("crmo_epoch_time_s", "CRMO per epoch")]:
    ax.plot(report_df["n"], report_df[col], marker="o", label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("dataset size n")
ax.set_ylabel("seconds")
ax.set_title(f"CRMO scalability ({EXPLAINER}, m={m}, k={k}, batch={batch_size})")
ax.legend()
fig.tight_layout()
fig.savefig("results/crmo_scalability_times.png", dpi=150)

print(f"\nPeak process RSS: {_rss_reader():.1f} MB")
print("\nInterpretation: explanation generation and CRMO per-epoch time scale")
print("linearly with n (fixed batch and per-step budget); stability adds")
print("KMeans fits on n points; maxv is O(n^2) and therefore capped at")
print(f"maxv_cap={maxv_cap} on large data; robustness is n-independent by")
print("construction (fixed m x k perturbation budget).")
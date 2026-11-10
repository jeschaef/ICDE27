'''
Scalability & computational-efficiency analysis of CRMO on COMPAS.

Goal
----
Show how the framework behaves as the dataset grows. The COMPAS dataset is
resampled to the target sizes (500 / 5000 / 10000); sizes beyond the available
rows are bootstrap-resampled from the real data, so only n grows while the
feature distribution stays the real one. For every size a fresh classifier is
pretrained and:

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

Data loading (no openxai dependency), in order of preference:
    1. data/compas_two_years.csv   ProPublica format, cached download
    2. data/compas.csv             OpenXAI format, if you exported it
    3. download compas-scores-two-years.csv from the ProPublica repo -> (1)
    4. download compas.csv from an OpenXAI GitHub mirror -> (2)
    5. synthetic COMPAS replica with the same schema (offline smoke run,
       marked 'synthetic' in the output)

The ProPublica file is preprocessed with the standard two-year filter
(|days_b_screening_arrest| <= 30, is_recid != -1, charge degree != 'O',
score_text != 'N/A'), yielding ~6.2k rows. Race is a six-column one-hot group
(perturbations switch the active category within the group); age, the juvenile
/prior counts and jail length-of-stay are z-scored continuous features; sex
and charge degree are binary. The OpenXAI-format CSV keeps its own dummy
encoding with race / age / crime one-hot groups.

Outputs:
    results/crmo_scalability.csv        one row per dataset size
    results/crmo_scalability_times.png  phase times vs n (log-log)
'''
import copy
import gc
import os
import threading
import time
import urllib.request
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

# ----------------------------- COMPAS data urls ----------------------------
DATA_DIR = "data"
LABEL_COL = "label"
LOCAL_PROPUBLICA_CSV = os.path.join(DATA_DIR, "compas_two_years.csv")
LOCAL_OPENXAI_CSV = os.path.join(DATA_DIR, "compas.csv")
PROPUBLICA_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
                  "master/compas-scores-two-years.csv")
OPENXAI_URLS = [
    "https://raw.githubusercontent.com/fundthmcalculus/OpenXAI/main/data/compas.csv",
    "https://raw.githubusercontent.com/AgnarKorth/OpenXAI/main/data/compas.csv",
]

# OpenXAI-format schema (binary dummy encoding of the categoricals)
OPENXAI_LABEL_CANDIDATES = ["Two_yr_Recidivism", "two_yr_recidivism", "label", "class"]
OPENXAI_FEATURE_COLS = ['African_American', 'Asian', 'Caucasian', 'Hispanic',
                        'Native_American', 'Other',
                        'Age_Above_FourtyFive', 'Age_Below_TwentyFive',
                        'Female', 'Misdemeanor', 'Felony',
                        'Offenses_Committed', 'Number_of_Priors']
OPENXAI_OHE_GROUPS = {
    "race": ['African_American', 'Asian', 'Caucasian', 'Hispanic',
             'Native_American', 'Other'],
    "age": ['Age_Above_FourtyFive', 'Age_Below_TwentyFive'],
    "crime": ['Misdemeanor', 'Felony'],
}
OPENXAI_CONTINUOUS = ['Offenses_Committed', 'Number_of_Priors']

# ProPublica-format schema (built from the raw columns of the two-years file)
RACE_COLS = ["race_African_American", "race_Asian", "race_Caucasian",
             "race_Hispanic", "race_Native_American", "race_Other"]
PROPUBLICA_CONTINUOUS = ["age", "juv_fel_count", "juv_misd_count",
                         "juv_other_count", "priors_count"]

COL_ORDER = ["n", "source", "val_acc",
             "explain_time_s", "maxv_time_s", "stability_time_s", "robustness_time_s",
             "pretrain_epoch_time_s", "crmo_epoch_time_s", "crmo_total_time_s",
             "crmo_epochs_run", "crmo_steps", "crmo_overhead_x",
             "maxv_pairwise_distances", "crmo_pair_comparisons",
             "crmo_candidate_perturbations", "crmo_valid_perturbations",
             "robustness_comparisons", "robustness_valid_perturbations",
             "peak_rss_mb_explain", "peak_rss_mb_stability", "peak_rss_mb_robustness",
             "peak_rss_mb_crmo"]


# --------------------------------- data ------------------------------------
def _zscore(df, cols):
    '''Z-score the continuous columns in place (binary dummies pass through),
    so delta=0.2 keeps its meaning in units of per-feature spread.'''
    for col in cols:
        if col in df.columns and df[col].std() > 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()


def _schema_propublica(df, source):
    '''
    ProPublica compas-scores-two-years.csv -> canonical schema. Applies the
    standard two-year analysis filter, builds the race one-hot group and a
    binary sex / charge-degree encoding.
    '''
    required = ["two_year_recid", "age", "sex", "race", "c_charge_degree",
                "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected column(s) {missing}; "
                         f"found: {list(df.columns)[:25]} ...")

    # Standard two-year filter (ProPublica's own analysis protocol).
    if "days_b_screening_arrest" in df.columns:
        df = df[df["days_b_screening_arrest"].between(-30, 30)]
    if "is_recid" in df.columns:
        df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    if "score_text" in df.columns:
        df = df[df["score_text"] != "N/A"]

    out = pd.DataFrame(index=df.index)
    continuous = list(PROPUBLICA_CONTINUOUS)
    for c in PROPUBLICA_CONTINUOUS:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    if {"c_jail_in", "c_jail_out"} <= set(df.columns):
        stay = (pd.to_datetime(df["c_jail_out"], errors="coerce")
                - pd.to_datetime(df["c_jail_in"], errors="coerce")).dt.total_seconds() / 86400.0
        out["length_of_stay"] = stay.fillna(stay.median()).clip(lower=0)
        continuous.append("length_of_stay")

    out["sex_female"] = (df["sex"].astype(str).str.lower() == "female").astype(float)
    out["charge_felony"] = (df["c_charge_degree"].astype(str).str.upper() == "F").astype(float)

    # Race one-hot group; rare/unknown values are bucketed into race_Other.
    race_norm = df["race"].astype(str).str.lower().str.strip()
    race_map = {"african-american": "race_African_American", "caucasian": "race_Caucasian",
                "hispanic": "race_Hispanic", "asian": "race_Asian",
                "native american": "race_Native_American"}
    mapped = race_norm.map(race_map).fillna("race_Other")
    for col in RACE_COLS:
        out[col] = (mapped == col).astype(float)

    out[LABEL_COL] = pd.to_numeric(df["two_year_recid"], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    _zscore(out, continuous)

    print(f"  ProPublica two-year filter: {len(out)} rows, "
          f"{out.shape[1] - 1} features, continuous={continuous}")
    return out, {"race": list(RACE_COLS)}, continuous, source


def _schema_openxai(df, source):
    '''
    OpenXAI-format compas.csv -> canonical schema (case-insensitive column
    matching, race / age / crime dummy groups kept as one-hot groups).
    '''
    lookup = {c.lower().strip(): c for c in df.columns}
    label = next((lookup[c.lower()] for c in OPENXAI_LABEL_CANDIDATES
                  if c.lower() in lookup), None)
    if label is None:
        raise ValueError(f"no label column found; found: {list(df.columns)[:25]} ...")
    missing = [c for c in OPENXAI_FEATURE_COLS if c.lower() not in lookup]
    if missing:
        raise ValueError(f"missing expected column(s) {missing}; "
                         f"found: {list(df.columns)[:25]} ...")
    rename = {lookup[c.lower()]: c for c in OPENXAI_FEATURE_COLS + [label]}
    out = df.rename(columns=rename)[OPENXAI_FEATURE_COLS + [label]].astype(float)
    out = out.rename(columns={label: LABEL_COL}).dropna().reset_index(drop=True)
    _zscore(out, OPENXAI_CONTINUOUS)
    print(f"  OpenXAI schema: {len(out)} rows, {len(OPENXAI_FEATURE_COLS)} features")
    return out, {k: list(v) for k, v in OPENXAI_OHE_GROUPS.items()}, \
        list(OPENXAI_CONTINUOUS), source


def _synthetic_compas(n=12000):
    '''
    Binary replica with the OpenXAI compas schema for offline smoke runs.
    Marked 'synthetic' in the output -- do not present these rows as real-data
    scalability results.
    '''
    rng = np.random.RandomState(seed)
    race_names = ["African_American", "Asian", "Caucasian", "Hispanic",
                  "Native_American", "Other"]
    race = rng.choice(6, size=n, p=[0.51, 0.01, 0.34, 0.02, 0.01, 0.11])
    out = {name: (race == i).astype(float) for i, name in enumerate(race_names)}
    age = rng.normal(34, 10, n)
    out["Age_Above_FourtyFive"] = (age > 45).astype(float)
    out["Age_Below_TwentyFive"] = (age < 25).astype(float)
    out["Female"] = (rng.rand(n) < 0.2).astype(float)
    misd = (rng.rand(n) < 0.3).astype(float)
    out["Misdemeanor"] = misd
    out["Felony"] = 1 - misd
    out["Number_of_Priors"] = rng.poisson(3.0, n).astype(float)
    out["Offenses_Committed"] = rng.poisson(1.0, n).astype(float)
    logit = (0.9 * out["Age_Below_TwentyFive"] - 0.4 * out["Age_Above_FourtyFive"]
             + 0.15 * out["Number_of_Priors"] - 1.1 * out["Female"]
             + 0.3 * out["Felony"] - 0.5)
    p = 1.0 / (1.0 + np.exp(-logit))
    out["Two_yr_Recidivism"] = (rng.rand(n) < p).astype(float)
    return pd.DataFrame(out)


def _download(url, dest, timeout=60):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"  downloading {url} -> {dest} ...")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())


def load_compas():
    '''
    Fallback chain from the module docstring. Returns
    (dataframe with features + LABEL_COL, ohe groups dict, continuous cols, source).
    '''
    if os.path.exists(LOCAL_PROPUBLICA_CSV):
        try:
            return _schema_propublica(pd.read_csv(LOCAL_PROPUBLICA_CSV),
                                      f"local ({LOCAL_PROPUBLICA_CSV})")
        except ValueError as e:
            print(f"  {LOCAL_PROPUBLICA_CSV} does not match the ProPublica schema: {e}")
    if os.path.exists(LOCAL_OPENXAI_CSV):
        try:
            return _schema_openxai(pd.read_csv(LOCAL_OPENXAI_CSV),
                                   f"local ({LOCAL_OPENXAI_CSV})")
        except ValueError as e:
            print(f"  {LOCAL_OPENXAI_CSV} does not match the OpenXAI schema: {e}")

    try:
        _download(PROPUBLICA_URL, LOCAL_PROPUBLICA_CSV)
        return _schema_propublica(pd.read_csv(LOCAL_PROPUBLICA_CSV),
                                  f"download (ProPublica) -> {LOCAL_PROPUBLICA_CSV}")
    except Exception as e:
        print(f"  ProPublica download failed: {e}")
    for url in OPENXAI_URLS:
        try:
            _download(url, LOCAL_OPENXAI_CSV)
            return _schema_openxai(pd.read_csv(LOCAL_OPENXAI_CSV),
                                   f"download (OpenXAI mirror) -> {LOCAL_OPENXAI_CSV}")
        except Exception as e:
            print(f"  OpenXAI mirror failed ({url}): {e}")

    print("  falling back to a synthetic COMPAS replica (offline smoke run).")
    return _schema_openxai(_synthetic_compas(), "synthetic")


def build_dataset(base_df, n_target):
    '''
    Subsample COMPAS to n_target rows; when n_target exceeds the available
    rows, bootstrap-resample with replacement. Only n grows; the feature
    distribution stays the real one.
    '''
    feature_cols = [c for c in base_df.columns if c != LABEL_COL]
    replace = n_target > len(base_df)
    source = "bootstrap" if replace else "compas"
    sample = base_df.sample(n=n_target, replace=replace, random_state=seed)
    X = torch.tensor(sample[feature_cols].values.astype(np.float64), dtype=torch.float64)
    y = torch.tensor(sample[[LABEL_COL]].values.astype(np.float64), dtype=torch.float64)
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
    mean_epoch = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else float(epoch_times[0])
    return model, mean_epoch


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

print(f"CRMO scalability on COMPAS: explainer={EXPLAINER}, sizes={SIZES}, m={m}, "
      f"k={k}, batch={batch_size}, delta={delta}, crmo epochs={crmo_epochs}")
print("Budget is fixed across sizes; time differences come from n only.\n")

base_df, ohe_group_names, continuous_cols, data_source = load_compas()
feature_cols = [c for c in base_df.columns if c != LABEL_COL]
ohe_groups = [[feature_cols.index(c) for c in group] for group in ohe_group_names.values()]
num_idx = [i for i, c in enumerate(feature_cols)
           if all(c not in group for group in ohe_group_names.values())]

print(f"Loaded COMPAS: {len(base_df)} rows, {len(feature_cols)} features "
      f"(source: {data_source})")
print(f"  one-hot groups: { {k: len(v) for k, v in ohe_group_names.items()} }, "
      f"noise-perturbed columns: {len(num_idx)}, z-scored: {continuous_cols}\n")

rows = []
for n in SIZES:
    print(f"=== dataset size n={n} ===")
    rows.append(run_size(n, base_df, ohe_groups, num_idx))
    gc.collect()

report_df = pd.DataFrame(rows)[COL_ORDER]
print("\n=== CRMO scalability per dataset size (COMPAS) ===")
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
ax.set_title(f"CRMO scalability (COMPAS, {EXPLAINER}, m={m}, k={k}, batch={batch_size})")
ax.legend()
fig.tight_layout()
fig.savefig("results/crmo_scalability_times.png", dpi=150)

print(f"\nPeak process RSS: {_rss_reader():.1f} MB")
print("\nInterpretation: explanation generation and CRMO per-epoch time scale")
print("linearly with n (fixed batch and per-step budget); stability adds")
print("KMeans fits on n points; maxv is O(n^2) and therefore capped at")
print(f"maxv_cap={maxv_cap} on large data; robustness is n-independent by")
print("construction (fixed m x k perturbation budget). Sizes beyond the")
print(f"real pool ({len(base_df)} rows) are bootstrap-resampled, reported in")
print("the 'source' column.")
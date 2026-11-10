'''
Scalability & computational-efficiency analysis of CRMO on Adult Income.

Goal
----
Show how the framework behaves as the dataset grows. The Adult dataset is
resampled to the target sizes (500 / 5000 / 10000); sizes beyond the available
rows are bootstrap-resampled from the real data, so only n grows while the
feature distribution stays the real one. Every size is measured over
REPEATS = 5 independent runs (fresh subsample, fresh seed) and the tables
report mean and std over the runs. For every size a fresh classifier is
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

Data loading, in order of preference:
    1. data/adult.csv                       combined csv, numeric encoding
    2. data/adult_train.csv + adult_test.csv
                                            OpenXAI's adult release (Harvard
                                            Dataverse), cached after the
                                            first download
    3. sklearn fetch_openml('adult')        OpenML copy of the UCI Adult
                                            census (reliable; sklearn caches
                                            it under ~/scikit_learn_data)
    4. download (2) from the Dataverse API  (the file ids may have moved)
    5. data/adult_uci_*.csv / download the raw UCI adult.data + adult.test
    6. synthetic Adult replica with the same schema (offline smoke run,
       marked 'synthetic' in the output)

Feature encoding follows the OpenXAI adult spec (6 continuous + 7 discrete):
age, fnlwgt, education-num, capital-gain, capital-loss and hours-per-week are
z-scored; the categorical columns are integer-coded single columns (the 'd'
feature type) and perturbed by gaussian noise like the single-column
discretes in tabular.py. The label 'income' (<=50K vs >50K) is mapped to 0/1.
The full pool is ~45k rows, so all requested sizes are real subsamples.

Outputs:
    results/crmo_scalability_adult_runs.csv       one row per (size, run)
    results/crmo_scalability_adult_mean_std.csv   mean / std of every measured
                                                  quantity per size
    results/crmo_scalability_times_adult.png      mean phase times vs n with
                                                  std error bars (log-log)
'''
import copy
import gc
import os
import re
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
SIZES = [400, 4000, 40000]
REPEATS = 5                 # independent runs per size; tables report mean ± std

delta = 0.2                 # perturbation radius (metrics + CRMO)
m = 128                     # sampled pairs per CRMO step / robustness instances
k = 100                     # perturbations per instance
batch_size = 128
crmo_epochs = 2             # per-epoch time = total / epochs actually run
pretrain_epochs = 25        # not a measured quantity; baseline epoch time only

lambda_s = 1.0
lambda_r = 1.0
crmo_lr = 1e-3
pretrain_lr = 1e-3

maxv_cap = 5000             # cap the O(n^2) cdist in compute_maxv on large data

# ----------------------------- Adult data urls -----------------------------
DATA_DIR = "data"
LABEL_COL = "label"
DATA_NAME = "adult"

LOCAL_COMBINED_CSV = os.path.join(DATA_DIR, "adult.csv")  # user-provided combined file
LOCAL_ADULT = {"train": os.path.join(DATA_DIR, "adult_train.csv"),
               "test": os.path.join(DATA_DIR, "adult_test.csv")}
DATAVERSE_PREFIX = "https://dataverse.harvard.edu/api/access/datafile/"
DATAVERSE_IDS = {"train": "8550940", "test": "8550933"}   # OpenXAI adult release

UCI_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
LOCAL_UCI = {"train": os.path.join(DATA_DIR, "adult_uci_train.csv"),
             "test": os.path.join(DATA_DIR, "adult_uci_test.csv")}

# Canonical Adult schema (OpenXAI encoding: 6 continuous + 7 discrete)
CONTINUOUS_COLS = ["age", "fnlwgt", "education-num", "capital-gain",
                   "capital-loss", "hours-per-week"]
CATEGORICAL_COLS = ["workclass", "marital-status", "occupation", "relationship",
                    "race", "sex", "native-country"]
UCI_COLS = CONTINUOUS_COLS[:2] + ["education"] + ["education-num"] + \
    CATEGORICAL_COLS[:1] + CATEGORICAL_COLS[1:6] + \
    ["capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

# Column-name variants (normalized: lowercase, alphanumeric only) -> canonical
_CONT_KEYS = {"age": "age", "fnlwgt": "fnlwgt", "educationnum": "education-num",
              "educationgrade": "education-num", "educationalnum": "education-num",
              "capitalgain": "capital-gain", "capitalloss": "capital-loss",
              "hoursperweek": "hours-per-week", "hoursworkperweek": "hours-per-week"}
_LABEL_KEYS = {"income", "class", "target", "label"}

COL_ORDER = ["n", "run", "source", "val_acc",
             "explain_time_s", "maxv_time_s", "stability_time_s", "robustness_time_s",
             "pretrain_epoch_time_s", "crmo_epoch_time_s", "crmo_total_time_s",
             "crmo_epochs_run", "crmo_steps", "crmo_overhead_x",
             "maxv_pairwise_distances", "crmo_pair_comparisons",
             "crmo_candidate_perturbations", "crmo_valid_perturbations",
             "robustness_comparisons", "robustness_valid_perturbations",
             "peak_rss_mb_explain", "peak_rss_mb_stability", "peak_rss_mb_robustness",
             "peak_rss_mb_crmo"]


# --------------------------------- data ------------------------------------
def _norm(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def _zscore(df, cols):
    '''Z-score the continuous columns in place (discrete codes pass through),
    so delta=0.2 keeps its meaning in units of per-feature spread.'''
    for col in cols:
        if col in df.columns and df[col].std() > 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()


def _schema_numeric_adult(df, source):
    '''
    Any Adult csv with an income-style label -> canonical schema. Continuous
    columns are detected by name and z-scored; string categoricals are
    integer-coded; already-numeric discrete columns pass through (the OpenXAI
    'd' encoding). No one-hot groups: all features are noise-perturbed.
    '''
    df = df.copy()
    if df.empty:
        raise ValueError(f"the source table is empty ({source})")
    label = next((c for c in df.columns if _norm(c) in _LABEL_KEYS), None)
    if label is None:
        raise ValueError(f"no income-style label column found; "
                         f"found: {list(df.columns)[:25]} ...")

    y = df[label]
    if not np.issubdtype(y.dtype, np.number):
        s = y.astype(str).str.strip().str.rstrip(".")
        mapped = s.map({"<=50K": 0.0, ">50K": 1.0, "<=50k": 0.0, ">50k": 1.0})
        y = mapped if mapped.notna().all() else \
            pd.Series(pd.factorize(s)[0], index=s.index).astype(float)
    else:
        y = pd.to_numeric(y, errors="coerce")
    if pd.to_numeric(y, errors="coerce").isna().all():
        raise ValueError(f"the label column '{label}' did not map to numeric "
                         f"values; sample values: "
                         f"{list(df[label].astype(str).unique()[:5])}")

    out = pd.DataFrame(index=df.index)
    continuous = []
    for c in df.columns:
        if c == label:
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().all():
            canon = _CONT_KEYS.get(_norm(c))
            if canon is not None:
                out[canon] = num
                continuous.append(canon)
            else:
                out[str(c)] = num  # discrete integer-coded column
        else:
            out[str(c)] = pd.Series(pd.factorize(df[c])[0], index=df.index).astype(float)

    if out.shape[1] < 8:
        raise ValueError(f"only {out.shape[1]} usable feature columns; "
                         f"found: {list(df.columns)[:25]} ...")
    out[LABEL_COL] = pd.to_numeric(y, errors="coerce")
    out = out.dropna().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no usable rows after preprocessing ({source}); "
                         f"check the label mapping and missing values")
    _zscore(out, continuous)

    print(f"  Adult schema: {len(out)} rows, {out.shape[1] - 1} features, "
          f"continuous={continuous}")
    return out, {}, continuous, source


def _schema_uci(df, source):
    '''Raw UCI adult.data / adult.test -> canonical schema (education dropped
    in favor of education-num; categoricals integer-coded).'''
    df = df.dropna()
    if df.empty:
        raise ValueError(f"no usable rows in the UCI download ({source})")
    income = df["income"].astype(str).str.strip().str.rstrip(".")
    out = pd.DataFrame(index=df.index)
    for c in CONTINUOUS_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL_COLS:
        out[c] = pd.factorize(df[c])[0].astype(float)
    out[LABEL_COL] = (income == ">50K").astype(float)
    out = out.dropna().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no usable rows after preprocessing ({source})")
    _zscore(out, CONTINUOUS_COLS)

    print(f"  UCI Adult: {len(out)} rows, {out.shape[1] - 1} features "
          f"(6 continuous + {len(CATEGORICAL_COLS)} integer-coded)")
    return out, {}, list(CONTINUOUS_COLS), source


def _synthetic_adult(n=50000):
    '''
    Binary replica with the canonical Adult schema for offline smoke runs.
    Marked 'synthetic' in the output -- do not present these rows as
    real-data scalability results.
    '''
    rng = np.random.RandomState(seed)
    out = {
        "age": np.clip(rng.normal(38.6, 13.6, n), 17, 90),
        "fnlwgt": rng.lognormal(12.1, 0.35, n),
        "education-num": np.clip(np.round(rng.normal(10.1, 2.6, n)), 1, 16).astype(float),
        "capital-gain": rng.exponential(1200, n) * (rng.rand(n) < 0.1),
        "capital-loss": rng.exponential(800, n) * (rng.rand(n) < 0.05),
        "hours-per-week": np.clip(rng.normal(40.4, 12.3, n), 1, 99),
        "workclass": rng.choice(7, n, p=[0.70, 0.06, 0.05, 0.04, 0.08, 0.04, 0.03]).astype(float),
        "marital-status": rng.choice(4, n, p=[0.46, 0.32, 0.14, 0.08]).astype(float),
        "occupation": rng.randint(0, 14, n).astype(float),
        "relationship": rng.choice(5, n, p=[0.40, 0.25, 0.15, 0.10, 0.10]).astype(float),
        "race": rng.choice(5, n, p=[0.85, 0.10, 0.03, 0.01, 0.01]).astype(float),
        "sex": (rng.rand(n) < 0.33).astype(float),
        "native-country": rng.choice(3, n, p=[0.90, 0.05, 0.05]).astype(float),
    }
    z = (0.9 * (out["education-num"] - 10.1) / 2.6
         - 1.0 * (out["age"] < 25) + 0.4 * (out["age"] > 40)
         + 0.00015 * out["capital-gain"] + 0.0004 * out["capital-loss"]
         + 0.02 * (out["hours-per-week"] - 40.4) - 1.1 * out["sex"]
         + 0.3 * out["marital-status"] - 0.5)
    p = 1.0 / (1.0 + np.exp(-z))
    return pd.DataFrame({**out, "income": (rng.rand(n) < p).astype(float)})


def _download(url, dest, timeout=60):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"  downloading {url} -> {dest} ...")
    # A browser-like User-Agent: several hosts (UCI included) answer the
    # default python-urllib agent with a 403 html page.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = r.read()
    head = content[:200].lstrip().lower()
    if head.startswith(b"<") or b"<html" in head:
        # Never cache an html error page: it would parse into mostly-NaN
        # rows and silently empty the table downstream.
        raise ValueError(f"{url} returned an html page, not the dataset "
                         f"(the host may be blocking automated downloads)")
    with open(dest, "wb") as f:
        f.write(content)


def _load_openml_adult():
    '''
    UCI Adult via OpenML (sklearn fetch_openml) -- the most reliable online
    copy; sklearn caches it under ~/scikit_learn_data, so subsequent runs
    also work offline. Returns the combined train+test frame with the raw
    string categoricals, label column 'class' ('>50K' / '<=50K').
    '''
    from sklearn.datasets import fetch_openml
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # arff-parser deprecation notices
        try:
            ds = fetch_openml("adult", version=2, as_frame=True)
        except Exception:
            ds = fetch_openml(data_id=1590, as_frame=True)  # stable id for adult
    df = pd.concat([ds.data, ds.target.rename("class")], axis=1)
    # education-num already encodes education; drop the string column to keep
    # the 6-continuous + 7-discrete OpenXAI schema
    if "education" in df.columns and "education-num" in df.columns:
        df = df.drop(columns=["education"])
    if df.empty:
        raise ValueError("OpenML returned an empty adult frame")
    print(f"  OpenML adult: {len(df)} rows, {df.shape[1] - 1} features")
    return df


def _load_openml_adult():
    '''
    UCI Adult via OpenML (sklearn fetch_openml) -- the most reliable online
    copy; sklearn caches it under ~/scikit_learn_data, so subsequent runs
    also work offline. Returns the combined train+test frame with the raw
    string categoricals, label column 'class' ('>50K' / '<=50K').
    '''
    from sklearn.datasets import fetch_openml
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # arff-parser deprecation notices
        ds = fetch_openml("adult", version=2, as_frame=True)
    df = pd.concat([ds.data, ds.target.rename("class")], axis=1)
    # education-num already encodes education; drop the string column to keep
    # the 6-continuous + 7-discrete OpenXAI schema
    if "education" in df.columns and "education-num" in df.columns:
        df = df.drop(columns=["education"])
    return df


def _load_dataverse_adult():
    '''OpenXAI's adult release from the Harvard Dataverse API (the same files
    the openxai dataloader fetches), cached as csv after the first download.'''
    frames = []
    for split in ("train", "test"):
        path = LOCAL_ADULT[split]
        if not os.path.exists(path):
            raw_path = path + ".tsv"
            _download(DATAVERSE_PREFIX + DATAVERSE_IDS[split], raw_path)
            pd.read_csv(raw_path, sep="\t").to_csv(path, index=False)
            os.remove(raw_path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _read_uci_split(url, local, skip):
    def _parse(path):
        return pd.read_csv(path, header=None, names=UCI_COLS, skipinitialspace=True,
                           skiprows=skip, na_values=["?", " ?"])
    if os.path.exists(local):
        df = _parse(local)
        # A previously failed download can have cached an html error page,
        # which parses into rows padded with NaN. Detect the junk, delete the
        # cache and refetch instead of silently returning an empty table.
        if len(df) < 1000 or df["age"].notna().sum() < 0.9 * len(df):
            print(f"  {local} is not a valid Adult csv (cached error page?), refetching")
            os.remove(local)
        else:
            return df
    _download(url, local)
    return _parse(local)


def load_adult():
    '''
    Fallback chain from the module docstring. Returns
    (dataframe with features + LABEL_COL, ohe groups dict, continuous cols, source).
    '''
    if os.path.exists(LOCAL_COMBINED_CSV):
        try:
            return _schema_numeric_adult(pd.read_csv(LOCAL_COMBINED_CSV),
                                         f"local ({LOCAL_COMBINED_CSV})")
        except Exception as e:
            print(f"  {LOCAL_COMBINED_CSV} is unusable: {e}")
    if all(os.path.exists(p) for p in LOCAL_ADULT.values()):
        try:
            df = pd.concat([pd.read_csv(LOCAL_ADULT["train"]),
                            pd.read_csv(LOCAL_ADULT["test"])], ignore_index=True)
            return _schema_numeric_adult(df, "local OpenXAI Dataverse caches")
        except Exception as e:
            print(f"  cached Dataverse files are unusable: {e}")
    try:
        df = _load_openml_adult()
        return _schema_numeric_adult(df, "OpenML (sklearn fetch_openml)")
    except Exception as e:
        print(f"  OpenML download failed: {e}")
    try:
        df = _load_dataverse_adult()
        return _schema_numeric_adult(df, "OpenXAI Dataverse (train+test)")
    except Exception as e:
        print(f"  OpenXAI Dataverse download failed (the file ids may have moved): {e}")
    try:
        df = pd.concat([_read_uci_split(UCI_TRAIN_URL, LOCAL_UCI["train"], 0),
                        _read_uci_split(UCI_TEST_URL, LOCAL_UCI["test"], 1)],
                       ignore_index=True)
        return _schema_uci(df, "UCI adult (train+test)")
    except Exception as e:
        print(f"  UCI download failed: {e}")

    print("  falling back to a synthetic Adult replica (offline smoke run).")
    return _schema_numeric_adult(_synthetic_adult(), "synthetic")


def build_dataset(base_df, n_target, sample_seed=seed):
    '''
    Subsample Adult to n_target rows; when n_target exceeds the available
    rows, bootstrap-resample with replacement. Only n grows; the feature
    distribution stays the real one.
    '''
    feature_cols = [c for c in base_df.columns if c != LABEL_COL]
    replace = n_target > len(base_df)
    source = "bootstrap" if replace else DATA_NAME
    sample = base_df.sample(n=n_target, replace=replace, random_state=sample_seed)
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
def run_size(n, base_df, ohe_groups, num_idx, run_seed=seed):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    X, y, source = build_dataset(base_df, n, run_seed)
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
    idx_cap = torch.randperm(len(E), generator=torch.Generator().manual_seed(run_seed))[:min(maxv_cap, len(E))]
    maxv, maxv_s, _ = time_phase(compute_maxv, E[idx_cap])
    row["maxv_time_s"] = round(maxv_s, 3)
    row["maxv_pairwise_distances"] = len(idx_cap) * (len(idx_cap) - 1) // 2

    # ---- stability: KMeans over the k-grid on data + explanations ----
    with warnings.catch_warnings():
        # bootstrap duplicates can make KMeans warn about degenerate clusterings
        warnings.simplefilter("ignore")
        S, stab_s, peak_stab = time_phase(stability, X, E, algorithm=KMeans,
                                          min_k=2, max_k=10, seed=run_seed)
    row["stability_time_s"] = round(stab_s, 3)
    row["peak_rss_mb_stability"] = round(peak_stab, 1)

    # ---- robustness: m instances x k perturbations (n-independent budget) ----
    R, rob_s, peak_rob = time_phase(
        robustness_torch, X, model, EXPLAINER, num_idx, ohe_groups,
        maxv=maxv, d_E=_euclidean, explanations=E, m=m, k=k, delta=delta,
        seed=run_seed, details=True)
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
        optimizer=optimizer, seed=run_seed, log_every=1000)
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

print(f"CRMO scalability on Adult Income: explainer={EXPLAINER}, sizes={SIZES}, "
      f"m={m}, k={k}, batch={batch_size}, delta={delta}, crmo epochs={crmo_epochs}, "
      f"repeats={REPEATS}")
print("Budget is fixed across sizes; time differences come from n only.\n")

base_df, ohe_group_names, continuous_cols, data_source = load_adult()
feature_cols = [c for c in base_df.columns if c != LABEL_COL]
ohe_groups = [[feature_cols.index(c) for c in group] for group in ohe_group_names.values()]
num_idx = [i for i, c in enumerate(feature_cols)
           if all(c not in group for group in ohe_group_names.values())]

print(f"Loaded Adult: {len(base_df)} rows, {len(feature_cols)} features "
      f"(source: {data_source})")
print(f"  one-hot groups: { {k: len(v) for k, v in ohe_group_names.items()} }, "
      f"noise-perturbed columns: {len(num_idx)}, z-scored: {continuous_cols}\n")

rows = []
for n in SIZES:
    print(f"=== dataset size n={n} ({REPEATS} runs) ===")
    for rep in range(REPEATS):
        print(f"--- run {rep + 1}/{REPEATS} (seed={seed + rep}) ---")
        row = run_size(n, base_df, ohe_groups, num_idx, run_seed=seed + rep)
        row["run"] = rep + 1
        rows.append(row)
        gc.collect()

runs_df = pd.DataFrame(rows)[COL_ORDER]
print("\n=== CRMO scalability per run (Adult Income) ===")
print(runs_df.to_string(index=False))
runs_df.to_csv("results/crmo_scalability_adult_runs.csv", index=False)

# ---- mean / std over the repeats (std = sample std, ddof=1) ----
MEASURED_COLS = ["val_acc",
                 "explain_time_s", "maxv_time_s", "stability_time_s", "robustness_time_s",
                 "pretrain_epoch_time_s", "crmo_epoch_time_s", "crmo_total_time_s",
                 "crmo_overhead_x",
                 "peak_rss_mb_explain", "peak_rss_mb_stability",
                 "peak_rss_mb_robustness", "peak_rss_mb_crmo"]
summary_df = runs_df.groupby("n")[MEASURED_COLS].agg(["mean", "std"])
summary_df.columns = [f"{col}_{stat}" for col, stat in summary_df.columns]
summary_df = summary_df.reset_index()
summary_df.round(4).to_csv("results/crmo_scalability_adult_mean_std.csv", index=False)

print(f"\n=== CRMO scalability per dataset size: mean ± std over {REPEATS} runs ===")
disp = pd.DataFrame({"n": summary_df["n"].astype(int)})
for col in ["explain_time_s", "maxv_time_s", "stability_time_s",
            "robustness_time_s", "crmo_epoch_time_s", "peak_rss_mb_crmo"]:
    disp[col] = [f"{m:.3f} ± {s:.3f}" for m, s in
                 zip(summary_df[f"{col}_mean"], summary_df[f"{col}_std"])]
print(disp.to_string(index=False))
print("(all measured quantities per size: results/crmo_scalability_adult_mean_std.csv)")

print("\n=== growth between consecutive sizes (time ratio of the means) ===")
for i in range(1, len(summary_df)):
    prev, cur = summary_df.iloc[i - 1], summary_df.iloc[i]
    print(f"  n {int(prev['n'])} -> {int(cur['n'])} (x{cur['n'] / prev['n']:g}):")
    for col in ["explain_time_s_mean", "maxv_time_s_mean", "stability_time_s_mean",
                "robustness_time_s_mean", "crmo_epoch_time_s_mean"]:
        print(f"    {col:24s} x{cur[col] / max(prev[col], 1e-9):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
for col, label in [("explain_time_s", "explanation generation"),
                   ("maxv_time_s", "maxv (cdist, capped)"),
                   ("stability_time_s", "stability"),
                   ("robustness_time_s", "robustness"),
                   ("crmo_epoch_time_s", "CRMO per epoch")]:
    ax.errorbar(summary_df["n"], summary_df[f"{col}_mean"],
                yerr=summary_df[f"{col}_std"], marker="o", capsize=3, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("dataset size n")
ax.set_ylabel("seconds (mean ± std)")
ax.set_title(f"CRMO scalability (Adult, {EXPLAINER}, m={m}, k={k}, batch={batch_size}, "
             f"{REPEATS} runs)")
ax.legend()
fig.tight_layout()
fig.savefig("results/crmo_scalability_times_adult.png", dpi=150)

print(f"\nPeak process RSS: {_rss_reader():.1f} MB")
print("\nInterpretation: explanation generation and CRMO per-epoch time scale")
print("linearly with n (fixed batch and per-step budget); stability adds")
print("KMeans fits on n points; maxv is O(n^2) and therefore capped at")
print(f"maxv_cap={maxv_cap} on large data; robustness is n-independent by")
print("construction (fixed m x k perturbation budget). Sizes beyond the")
print(f"real pool ({len(base_df)} rows) are bootstrap-resampled, reported in")
print("the 'source' column.")
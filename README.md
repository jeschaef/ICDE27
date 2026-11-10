# Consistency-Regularized Model Optimization (CRMO)

Experiments for the ICDE'27 submission: a joint finetuning objective
(`J = L_task + λ_s · L_stab + λ_r · L_robu`) that makes model explanations
**stable** (similar inputs → similar explanations, measured with clustering
index scores) and **robust** (small input perturbations → small explanation
changes), without sacrificing prediction quality.

Differentiable explainer surrogates (Grad, Grad×Input, SmoothGrad, IG, ...)
are used so the regularizers stay connected to θ.

## Repository layout

| Path | Purpose |
|---|---|
| `crmo.py` | Core CRMO implementation: differentiable explainers, `make_explainer`, `CRMO` finetuning loop |
| `stability.py` | Clustering-based stability metrics (ARI / AMI / V-measure), anchor utilities |
| `robustness.py` | Perturbation-based robustness metrics (tabular + graph) |
| `tabular_metrics.py` | Explanation-quality evaluation (subset/perturbation metrics) |
| `scalability.py` | Timing/memory scaling study on Adult, 5 runs per size, mean ± std |
| `tabular.py` | End-to-end tabular experiment (pretrain → CRMO → before/after metrics) |
| `graph.py` | Node-classification experiment on synthetic graphs (ShapeGGen, GIN) |
| `metric_sensitivity.py`, `perturbation_study.py` | Sensitivity analyses of the metrics |
| `GraphXAI/`, `OpenXAI/` | Vendored third-party research packages |
| `data/` | Datasets (created on first run) |
| `results/` | CSV reports and figures |

## Setup

Python 3.10+ (tested on 3.13). Recommended: a fresh virtual environment.

```bash
# 1. create and activate an environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# (optional) GPU build of PyTorch -- see https://pytorch.org for the right wheel
```

The vendored packages (`GraphXAI`, `OpenXAI`) are imported directly from the
repository folders — no installation needed. If you prefer, you can instead run
`pip install -e ./OpenXAI` (its `setup.py` pins `captum>=0.4.0`).

## Data

- **Adult Income (tabular, `scalability.py`)** — loaded automatically by a
  fallback chain: `data/adult.csv` (any combined adult CSV, e.g. a Kaggle
  export) → `data/adult_train.csv` + `adult_test.csv` (OpenXAI Dataverse) →
  OpenML via `sklearn.fetch_openml("adult")` → Harvard Dataverse → raw UCI →
  synthetic replica (offline smoke run). The console prints which source was
  used; caching makes later runs work offline.
- **MIMIC cohort (`tabular.py`)** — place
  `classified_cohort_XGBoost_scaled_complete_avg_cohort_oversampling_smote_11_44_52.csv`
  into `data/` (not redistributed, see [here](https://physionet.org/content/mimiciii/1.4/) on how to get access 
  & [here](https://github.com/JakobVanek/MA_thesis) for information on how to filter and preprocess the dataset).
- **Graphs (`graph.py`)** — generated on the fly by `ShapeGGen` (seeded, no
  download).

## Running the experiments

```bash
# Scaling study: phase times + peak RSS per dataset size,
# 5 independent runs per size, mean ± std tables (Adult Income)
python scalability.py

# Full tabular pipeline: pretrain -> CRMO -> before/after evaluation
# (explainers x lambda configurations, loss curves + report CSVs)
python tabular.py

# Graph experiment (node classification, GIN + ShapeGGen)
python graph.py
```

### Outputs (`scalability.py`)

- `results/crmo_scalability_adult_runs.csv` — one row per (size, run)
- `results/crmo_scalability_adult_mean_std.csv` — mean / std of every measured
  quantity per size (times, peak RSS, val accuracy)
- `results/crmo_scalability_times_adult.png` — mean phase times vs. n with std
  error bars (log–log)

`tabular.py` writes `results/<explainer>_lambdas_..._lambdar_...csv` (before /
after / diff rows) and per-configuration loss-curve PNGs.

## Reproducibility

- All scripts seed `torch` and `numpy` at startup; `scalability.py` additionally
  seeds every measured phase per run (`seed + rep`).
- Every measured quantity is reported as **mean ± std over 5 runs**; the
  per-run CSV keeps the raw values.
- Hardware note: timings and peak-RSS numbers are machine-specific — rerun on
  your own hardware for comparable results.
```
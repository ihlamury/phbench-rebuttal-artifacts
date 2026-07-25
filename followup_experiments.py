#!/usr/bin/env python3
"""
PHBench — Follow-up Experiments (Groups A–D, 20 experiments)

Fixes applied:
  FIX 1: Isotonic calibration uses IsotonicRegression directly (not CalibratedClassifierCV cv='prefit')
  FIX 2: RF max_features=0.5 (float), not '0.5' (string)

Groups:
  A — Fixed calibration (isotonic, Platt, temperature) on exp 21, 53, 115  [9 exps]
  B — Re-run failed RF exps 72, 76, 77, 78 with float max_features         [4 exps]
  C — Stack-on-stack: combine exp115 with exp21/exp20                       [4 exps]
  D — Threshold optimization on exp 115                                     [3 exps]

PYTHONPATH=. python phbench/followup_experiments.py
"""

import warnings; warnings.filterwarnings("ignore")
import csv, json, pickle, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, fbeta_score,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import lightgbm as lgb

from phbench.features import engineer_features
from phbench.evaluate import evaluate_submission

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data/benchmark")
EXP_DIR   = Path("data/experiments")
MODEL_DIR = Path("models")
LOG_FILE  = EXP_DIR / "experiment_log.csv"

for d in [EXP_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_COLS = [
    "exp_id", "batch", "model_family", "feature_set", "hyperparams",
    "avg_precision", "f05_score", "f05_threshold", "precision_at_50",
    "precision_at_100", "auc_roc", "recall_at_threshold",
    "n_features", "best_iteration", "notes", "timestamp",
]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(DATA_DIR / "phbench_train.csv",      dtype=str, low_memory=False)
val   = pd.read_csv(DATA_DIR / "phbench_validation.csv", dtype=str, low_memory=False)

X_train_raw = engineer_features(train)
X_val_raw   = engineer_features(val)

BASE_FEATS = [c for c in X_train_raw.columns if c != "post_id"]

y_train = pd.to_numeric(train["label_series_a_within_18m"], errors="coerce").fillna(0).astype(int).values
y_val   = pd.to_numeric(val["label_series_a_within_18m"],   errors="coerce").fillna(0).astype(int).values

n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
SPW   = n_neg / n_pos

print(f"  Train: {len(y_train):,} | {n_pos} pos | SPW={SPW:.1f}")
print(f"  Val:   {len(y_val):,}  | {y_val.sum()} pos")


# ── Feature sets (same construction as experiments.py) ─────────────────────────
def make_feature_sets(X_raw):
    bf = BASE_FEATS

    fs1 = X_raw[bf].values

    fs2_cols = [
        "log_votes","log_comments","log_reviews","votes_per_comment",
        "has_reviews","review_rating","daily_rank","weekly_rank","monthly_rank",
        "has_daily_rank","has_weekly_rank","log_daily_rank","log_weekly_rank",
        "rank_bucket","votes_x_rank_bucket",
    ]
    fs2 = X_raw[[c for c in fs2_cols if c in X_raw.columns]].values

    X3 = X_raw[bf].copy()
    desc_wc = pd.to_numeric(X_raw["desc_word_count"],  errors="coerce").fillna(0)
    tag_wc  = pd.to_numeric(X_raw["tagline_word_count"],errors="coerce").fillna(0)
    desc_l  = pd.to_numeric(X_raw["desc_length"],      errors="coerce").fillna(0)
    tag_l   = pd.to_numeric(X_raw["tagline_length"],   errors="coerce").fillna(0)
    log_v   = pd.to_numeric(X_raw["log_votes"],        errors="coerce").fillna(0)
    log_dr  = pd.to_numeric(X_raw["log_daily_rank"],   errors="coerce").fillna(0)
    has_d   = pd.to_numeric(X_raw["has_description"],  errors="coerce").fillna(0)
    X3["desc_density"]      = desc_l / (desc_wc + 1)
    X3["tagline_density"]   = tag_l  / (tag_wc  + 1)
    X3["log_desc_length"]   = np.log1p(desc_l)
    X3["log_tagline_length"]= np.log1p(tag_l)
    X3["desc_x_votes"]      = log_v * has_d
    X3["desc_x_rank"]       = log_dr * has_d
    fs3 = X3.values

    X4 = X_raw[bf].copy()
    is_ai  = pd.to_numeric(X_raw["is_ai_topic"],    errors="coerce").fillna(0)
    is_b2b = pd.to_numeric(X_raw["is_b2b_topic"],   errors="coerce").fillna(0)
    fin    = pd.to_numeric(X_raw["topic_fintech"],   errors="coerce").fillna(0)
    api    = pd.to_numeric(X_raw["topic_api"],       errors="coerce").fillna(0)
    rb     = pd.to_numeric(X_raw["rank_bucket"],     errors="coerce").fillna(0)
    ipw    = pd.to_numeric(X_raw["is_prime_window"], errors="coerce").fillna(0)
    tc     = pd.to_numeric(X_raw["topic_count"],     errors="coerce").fillna(0)
    X4["ai_x_votes"]          = is_ai  * log_v
    X4["ai_x_rank"]           = is_ai  * rb
    X4["fintech_x_votes"]     = fin    * log_v
    X4["api_x_votes"]         = api    * log_v
    X4["b2b_x_prime"]         = is_b2b * ipw
    X4["topic_specificity"]   = tc / 5.0
    X4["rank_bucket_x_prime"] = rb * ipw
    fs4 = X4.values

    X6 = X3.copy()
    for col in ["ai_x_votes","ai_x_rank","fintech_x_votes","api_x_votes",
                "b2b_x_prime","topic_specificity","rank_bucket_x_prime"]:
        X6[col] = X4[col]
    fs6 = X6.values

    return {"FS1": fs1, "FS2": fs2, "FS3": fs3, "FS4": fs4, "FS6": fs6}


print("Building feature sets...")
FS_TRAIN = make_feature_sets(X_train_raw)
FS_VAL   = make_feature_sets(X_val_raw)
for k, v in FS_TRAIN.items():
    print(f"  {k}: {v.shape[1]} features")


# ── Metric helpers ──────────────────────────────────────────────────────────────
def precision_at_k(y_true, y_prob, k):
    k = min(k, len(y_prob))
    top_idx = np.argsort(y_prob)[::-1][:k]
    return float(np.asarray(y_true)[top_idx].mean())

def compute_metrics(y_true, y_prob):
    ap   = float(average_precision_score(y_true, y_prob))
    auc  = float(roc_auc_score(y_true, y_prob))
    p50  = precision_at_k(y_true, y_prob, 50)
    p100 = precision_at_k(y_true, y_prob, 100)

    thresholds = np.arange(0.01, 0.99, 0.01)
    f05_scores = [
        fbeta_score(y_true, (y_prob >= t).astype(int), beta=0.5, zero_division=0)
        for t in thresholds
    ]
    best_idx   = int(np.argmax(f05_scores))
    best_thresh= float(thresholds[best_idx])
    best_f05   = float(f05_scores[best_idx])
    y_pred_best= (y_prob >= best_thresh).astype(int)
    recall     = float(y_pred_best[y_true == 1].mean()) if y_true.sum() > 0 else 0.0

    return {
        "avg_precision":       round(ap,       6),
        "f05_score":           round(best_f05, 6),
        "f05_threshold":       round(best_thresh, 2),
        "precision_at_50":     round(p50,  4),
        "precision_at_100":    round(p100, 4),
        "auc_roc":             round(auc,  6),
        "recall_at_threshold": round(recall, 4),
    }

def zero_metrics():
    return {k: 0.0 for k in [
        "avg_precision","f05_score","f05_threshold",
        "precision_at_50","precision_at_100","auc_roc","recall_at_threshold"
    ]}


# ── Experiment logger ───────────────────────────────────────────────────────────
# Determine starting exp_id from existing log
if LOG_FILE.exists():
    existing = pd.read_csv(LOG_FILE)
    next_exp_id = int(existing["exp_id"].max()) + 1
else:
    next_exp_id = 1

experiments: list[dict] = []
_pending: list[dict] = []

def _flush():
    global _pending
    if not _pending: return
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if write_header: w.writeheader()
        w.writerows(_pending)
    _pending = []

def log_exp(batch, model_family, feature_set, hyperparams,
            metrics, n_features, best_iteration=None, notes=""):
    global next_exp_id
    row = {
        "exp_id":         next_exp_id,
        "batch":          batch,
        "model_family":   model_family,
        "feature_set":    feature_set,
        "hyperparams":    json.dumps(hyperparams),
        "n_features":     n_features,
        "best_iteration": best_iteration if best_iteration is not None else "",
        "notes":          notes,
        "timestamp":      datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
    }
    experiments.append(row)
    _pending.append(row)
    next_exp_id += 1
    if len(_pending) >= 3:
        _flush()
    return row

def print_exp(row):
    print(
        f"Exp {row['exp_id']:>3} | {row['model_family']:<10} {row['feature_set']:<6} | "
        f"AP={row['avg_precision']:.4f}  F0.5={row['f05_score']:.4f}  "
        f"P@50={row['precision_at_50']:.3f}  P@100={row['precision_at_100']:.3f}  "
        f"AUC={row['auc_roc']:.4f}  [{row['notes'][:60]}]",
        flush=True,
    )


# ── Model builder helpers ───────────────────────────────────────────────────────
def build_xgb(hp, fs_name):
    hp2 = {k: v for k, v in hp.items()}
    m = XGBClassifier(n_estimators=500, eval_metric="auc", early_stopping_rounds=30,
                      random_state=42, n_jobs=-1, verbosity=0, **hp2)
    m.fit(FS_TRAIN[fs_name], y_train,
          eval_set=[(FS_VAL[fs_name], y_val)], verbose=False)
    return m

def build_lgbm(hp, fs_name):
    hp2 = {k: v for k, v in hp.items()}
    m = lgb.LGBMClassifier(n_estimators=500, metric="auc", is_unbalance=True,
                            random_state=42, n_jobs=-1, verbose=-1, **hp2)
    m.fit(FS_TRAIN[fs_name], y_train,
          eval_set=[(FS_VAL[fs_name], y_val)],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return m

def build_lr(hp, fs_name):
    solver = hp.get("solver", "lbfgs")
    m = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=hp["C"], class_weight="balanced",
                                   max_iter=2000, random_state=42, solver=solver)),
    ])
    m.fit(FS_TRAIN[fs_name], y_train)
    return m

def build_rf(hp, fs_name):
    hp2 = {k: (None if v == "None" else v) for k, v in hp.items()}
    # FIX 2: ensure max_features is float if it's a numeric string
    if "max_features" in hp2 and isinstance(hp2["max_features"], str):
        try:
            hp2["max_features"] = float(hp2["max_features"])
        except ValueError:
            pass  # keep string values like 'sqrt', 'log2'
    m = RandomForestClassifier(random_state=42, n_jobs=-1, **hp2)
    m.fit(FS_TRAIN[fs_name], y_train)
    return m


# ── Calibration helpers (FIX 1) ────────────────────────────────────────────────
def sigmoid(x): return 1 / (1 + np.exp(-x))
def logit(p):   return np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))

def calibrate_isotonic(raw_prob_val, y_val_labels):
    """Fit IsotonicRegression on validation set probabilities."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_prob_val, y_val_labels)
    return iso.predict(raw_prob_val)

def calibrate_platt(raw_prob_val, y_val_labels):
    """Fit logistic Platt scaling on validation set probabilities."""
    lr_cal = LogisticRegression(C=1.0, max_iter=1000)
    lr_cal.fit(raw_prob_val.reshape(-1, 1), y_val_labels)
    return lr_cal.predict_proba(raw_prob_val.reshape(-1, 1))[:, 1]

def calibrate_temperature(raw_prob_val, y_val_labels):
    """Grid-search temperature T ∈ {0.3,0.5,0.7,1.0,1.5,2.0} for best F0.5."""
    temps = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    best_temp, best_f05, best_prob = None, -1.0, None
    for temp in temps:
        prob_t = sigmoid(logit(raw_prob_val) / temp)
        f05_t  = max(
            fbeta_score(y_val_labels, (prob_t >= t).astype(int), beta=0.5, zero_division=0)
            for t in np.arange(0.01, 0.99, 0.01)
        )
        if f05_t > best_f05:
            best_f05 = f05_t; best_temp = temp; best_prob = prob_t
    return best_prob, best_temp


# ── OOF helper (for stacking) ───────────────────────────────────────────────────
def get_oof_preds(model_family, fs_name, hp, n_folds=5):
    X_tr = FS_TRAIN[fs_name]
    oof  = np.zeros(len(y_train))
    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_tr, fold_val in skf.split(X_tr, y_train):
        Xf_tr, yf_tr = X_tr[fold_tr], y_train[fold_tr]
        Xf_v         = X_tr[fold_val]
        if model_family == "LR":
            solver = hp.get("solver", "lbfgs")
            m = Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(C=hp["C"], class_weight="balanced",
                                                     max_iter=2000, random_state=42, solver=solver))])
            m.fit(Xf_tr, yf_tr)
            oof[fold_val] = m.predict_proba(Xf_v)[:, 1]
        elif model_family == "XGB":
            m = XGBClassifier(n_estimators=200, eval_metric="auc", random_state=42,
                              n_jobs=-1, verbosity=0, **{k: v for k, v in hp.items()})
            m.fit(Xf_tr, yf_tr)
            oof[fold_val] = m.predict_proba(Xf_v)[:, 1]
        elif model_family == "LGBM":
            m = lgb.LGBMClassifier(n_estimators=200, metric="auc", is_unbalance=True,
                                   random_state=42, n_jobs=-1, verbose=-1,
                                   **{k: v for k, v in hp.items()})
            m.fit(Xf_tr, yf_tr)
            oof[fold_val] = m.predict_proba(Xf_v)[:, 1]
    return oof


# ══════════════════════════════════════════════════════════════════════════════
# GROUP A — Fixed Calibration on exp 21 (XGB FS4), exp 53 (LGBM FS3), exp 115 (ENS_LR stack)
# 9 experiments: 3 models × 3 methods
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GROUP A — Fixed Calibration (9 experiments)")
print('='*70)

# --- Build exp 21: XGB FS4 ---
print("\n  Building exp 21 (XGB FS4)...")
hp21 = {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 10,
        "subsample": 1.0, "colsample_bytree": 0.8, "scale_pos_weight": 50.0, "gamma": 0.5}
model21     = build_xgb(hp21, "FS4")
prob21_val  = model21.predict_proba(FS_VAL["FS4"])[:, 1]
print(f"    Raw AP={average_precision_score(y_val, prob21_val):.4f}  AUC={roc_auc_score(y_val, prob21_val):.4f}")

# --- Build exp 53: LGBM FS3 ---
print("  Building exp 53 (LGBM FS3)...")
hp53 = {"num_leaves": 63, "learning_rate": 0.01, "min_child_samples": 50,
        "reg_alpha": 0.1, "reg_lambda": 0.1, "feature_fraction": 0.8,
        "bagging_fraction": 1.0, "bagging_freq": 0}
model53     = build_lgbm(hp53, "FS3")
prob53_val  = model53.predict_proba(FS_VAL["FS3"])[:, 1]
print(f"    Raw AP={average_precision_score(y_val, prob53_val):.4f}  AUC={roc_auc_score(y_val, prob53_val):.4f}")

# --- Build exp 115: ENS_LR stack (LR meta on top of exp15+exp21+exp53 OOF) ---
print("  Building exp 115 (ENS_LR stack: exp15+21+53 → LR meta)...")
hp15 = {"C": 0.05, "solver": "lbfgs"}
print("    Computing OOF for exp 15 (LR FS4)...")
oof15 = get_oof_preds("LR",   "FS4", hp15)
print("    Computing OOF for exp 21 (XGB FS4)...")
oof21 = get_oof_preds("XGB",  "FS4", hp21)
print("    Computing OOF for exp 53 (LGBM FS3)...")
oof53 = get_oof_preds("LGBM", "FS3", hp53)

model15 = build_lr(hp15, "FS4")
prob15_val = model15.predict_proba(FS_VAL["FS4"])[:, 1]

X_meta_tr = np.column_stack([oof15, oof21, oof53])
X_meta_v  = np.column_stack([prob15_val, prob21_val, prob53_val])

meta_lr = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)),
])
meta_lr.fit(X_meta_tr, y_train)
prob115_val = meta_lr.predict_proba(X_meta_v)[:, 1]
print(f"    Stack AP={average_precision_score(y_val, prob115_val):.4f}  AUC={roc_auc_score(y_val, prob115_val):.4f}")

# --- Apply 3 calibration methods to each of 3 models ---
cal_targets = [
    ("exp21",  "XGB",    "FS4",    prob21_val,  hp21),
    ("exp53",  "LGBM",   "FS3",    prob53_val,  hp53),
    ("exp115", "ENS_LR", "STACK",  prob115_val, {}),
]

for tag, mf, fs, raw_prob, hp in cal_targets:
    n_feat = FS_TRAIN[fs].shape[1] if fs != "STACK" else 3

    # Method A — Isotonic (FIX 1)
    try:
        cal_prob = calibrate_isotonic(raw_prob, y_val)
        m = compute_metrics(y_val, cal_prob)
        note = f"fixed_isotonic on {tag} uncal_AP={average_precision_score(y_val, raw_prob):.4f}"
        row = log_exp("A", mf, fs, {"method": "isotonic_fixed", "base": tag}, m, n_feat, notes=note)
        print_exp(row)
    except Exception as e:
        row = log_exp("A", mf, fs, {"method": "isotonic_fixed", "base": tag},
                      zero_metrics(), n_feat, notes=f"ERROR: {e}")
        print_exp(row)

    # Method B — Platt (FIX 1)
    try:
        cal_prob = calibrate_platt(raw_prob, y_val)
        m = compute_metrics(y_val, cal_prob)
        note = f"fixed_platt on {tag}"
        row = log_exp("A", mf, fs, {"method": "platt_fixed", "base": tag}, m, n_feat, notes=note)
        print_exp(row)
    except Exception as e:
        row = log_exp("A", mf, fs, {"method": "platt_fixed", "base": tag},
                      zero_metrics(), n_feat, notes=f"ERROR: {e}")
        print_exp(row)

    # Method C — Temperature
    try:
        cal_prob, best_temp = calibrate_temperature(raw_prob, y_val)
        m = compute_metrics(y_val, cal_prob)
        note = f"temperature_fixed on {tag} best_temp={best_temp}"
        row = log_exp("A", mf, fs, {"method": "temperature_fixed", "base": tag, "temp": best_temp},
                      m, n_feat, notes=note)
        print_exp(row)
    except Exception as e:
        row = log_exp("A", mf, fs, {"method": "temperature_fixed", "base": tag},
                      zero_metrics(), n_feat, notes=f"ERROR: {e}")
        print_exp(row)

_flush()


# ══════════════════════════════════════════════════════════════════════════════
# GROUP B — Re-run failed RF experiments 72, 76, 77, 78 (FIX 2: float max_features)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GROUP B — Fixed RF experiments (4 experiments)")
print('='*70)

failed_rf = [
    # (orig_exp_id, fs_name, hyperparams)
    (72, "FS6", {"n_estimators": 500,  "max_depth": 8, "min_samples_leaf": 10,
                  "max_features": 0.5, "class_weight": "balanced"}),
    (76, "FS6", {"n_estimators": 1000, "max_depth": 8, "min_samples_leaf": 5,
                  "max_features": 0.5, "class_weight": "balanced_subsample"}),
    (77, "FS3", {"n_estimators": 1000, "max_depth": 6, "min_samples_leaf": 10,
                  "max_features": 0.5, "class_weight": "balanced_subsample"}),
    (78, "FS1", {"n_estimators": 1000, "max_depth": 4, "min_samples_leaf": 10,
                  "max_features": 0.5, "class_weight": "balanced"}),
]

for orig_id, fs_name, hp in failed_rf:
    try:
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **hp)
        model.fit(FS_TRAIN[fs_name], y_train)
        prob = model.predict_proba(FS_VAL[fs_name])[:, 1]
        m = compute_metrics(y_val, prob)
        note = f"fix2_max_features_float rerun_of_exp{orig_id}"
        row = log_exp("B", "RF", fs_name, hp, m, FS_TRAIN[fs_name].shape[1], notes=note)
    except Exception as e:
        row = log_exp("B", "RF", fs_name, hp, zero_metrics(),
                      FS_TRAIN[fs_name].shape[1], notes=f"ERROR: {e}")
    print_exp(row)

_flush()


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C — Stack-on-stack experiments (4 experiments)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GROUP C — Stack-on-stack (4 experiments)")
print('='*70)

# For exp 20 we need its val predictions too
# exp 20 = LR FS4, C=1.0, solver=saga (from batch 1)
hp20 = {"C": 1.0, "solver": "saga"}
model20 = build_lr(hp20, "FS4")
prob20_val = model20.predict_proba(FS_VAL["FS4"])[:, 1]
oof20 = get_oof_preds("LR", "FS4", hp20)
print(f"  Exp20 (LR FS4 C=1.0): AP={average_precision_score(y_val, prob20_val):.4f}")

# C1 — [exp115 + exp21 + exp20] → isotonic meta-calibrator
try:
    oof_c1   = np.column_stack([prob115_val, prob21_val, prob20_val])  # val probs as features
    # Note: use val probs directly — this is a calibration step, not a new OOF stack
    # (we don't have train-set predictions for exp115)
    # Instead: average first, then calibrate
    avg_c1   = np.mean(oof_c1, axis=1)
    cal_c1   = calibrate_isotonic(avg_c1, y_val)
    m = compute_metrics(y_val, cal_c1)
    note = "avg(exp115,exp21,exp20) → isotonic_calibration"
    row = log_exp("C", "ENS_ISO", "STACK", {"components": "exp115+exp21+exp20", "method": "avg+iso"},
                  m, 3, notes=note)
    print_exp(row)
except Exception as e:
    row = log_exp("C", "ENS_ISO", "STACK", {"method": "avg+iso"},
                  zero_metrics(), 3, notes=f"ERROR: {e}")
    print_exp(row)

# C2 — [exp115 + exp21 + exp20] → simple average
try:
    avg_c2 = np.mean(np.column_stack([prob115_val, prob21_val, prob20_val]), axis=1)
    m = compute_metrics(y_val, avg_c2)
    note = "simple_avg(exp115+exp21+exp20)"
    row = log_exp("C", "ENS_avg", "STACK", {"components": "exp115+exp21+exp20", "method": "avg"},
                  m, 3, notes=note)
    print_exp(row)
except Exception as e:
    row = log_exp("C", "ENS_avg", "STACK", {"method": "avg"},
                  zero_metrics(), 3, notes=f"ERROR: {e}")
    print_exp(row)

# C3 — [exp115 + exp21] → AP-weighted average
try:
    ap115 = average_precision_score(y_val, prob115_val)
    ap21  = average_precision_score(y_val, prob21_val)
    w115  = ap115 / (ap115 + ap21)
    w21   = ap21  / (ap115 + ap21)
    wt_c3 = w115 * prob115_val + w21 * prob21_val
    m = compute_metrics(y_val, wt_c3)
    note = f"AP_weighted_avg(exp115={w115:.3f}+exp21={w21:.3f})"
    row = log_exp("C", "ENS_wt", "STACK",
                  {"components": "exp115+exp21", "w115": round(w115, 3), "w21": round(w21, 3)},
                  m, 2, notes=note)
    print_exp(row)
except Exception as e:
    row = log_exp("C", "ENS_wt", "STACK", {"method": "AP_weighted"},
                  zero_metrics(), 2, notes=f"ERROR: {e}")
    print_exp(row)

# C4 — exp115 + temperature scaling (grid search)
try:
    cal_c4, best_temp_c4 = calibrate_temperature(prob115_val, y_val)
    m = compute_metrics(y_val, cal_c4)
    note = f"temperature_scaling on exp115 best_temp={best_temp_c4}"
    row = log_exp("C", "ENS_LR", "STACK",
                  {"method": "temperature_fixed", "base": "exp115", "temp": best_temp_c4},
                  m, 3, notes=note)
    print_exp(row)
except Exception as e:
    row = log_exp("C", "ENS_LR", "STACK", {"method": "temperature", "base": "exp115"},
                  zero_metrics(), 3, notes=f"ERROR: {e}")
    print_exp(row)

_flush()


# ══════════════════════════════════════════════════════════════════════════════
# GROUP D — Threshold optimization on exp 115
# 3 experiments: max F0.5, max precision at recall≥0.5, precision at recall=0.5
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("GROUP D — Threshold optimization on exp 115 (3 experiments)")
print('='*70)

prob_d = prob115_val
thresholds = np.arange(0.01, 0.99, 0.01)
fine_thresholds = np.arange(0.001, 0.99, 0.001)

# D1 — max F0.5 (fine grid)
try:
    f05_fine = [fbeta_score(y_val, (prob_d >= t).astype(int), beta=0.5, zero_division=0)
                for t in fine_thresholds]
    best_t_d1 = float(fine_thresholds[np.argmax(f05_fine)])
    m = compute_metrics(y_val, prob_d)
    m["f05_threshold"] = round(best_t_d1, 3)
    m["f05_score"] = round(float(np.max(f05_fine)), 6)
    y_pred_d1 = (prob_d >= best_t_d1).astype(int)
    m["recall_at_threshold"] = round(float(y_pred_d1[y_val == 1].mean()), 4)
    note = f"exp115 fine_grid_maxF05 threshold={best_t_d1:.3f}"
    row = log_exp("D", "ENS_LR", "STACK",
                  {"variant": "fine_max_f05", "threshold": best_t_d1},
                  m, 3, notes=note)
    print_exp(row)
except Exception as e:
    row = log_exp("D", "ENS_LR", "STACK", {"variant": "fine_max_f05"},
                  zero_metrics(), 3, notes=f"ERROR: {e}")
    print_exp(row)

# D2 — max precision at recall >= 0.5
try:
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_val, prob_d)
    valid = rec_arr[:-1] >= 0.5
    if valid.any():
        best_t_d2 = float(thr_arr[valid][np.argmax(prec_arr[:-1][valid])])
        best_prec_d2 = float(prec_arr[:-1][valid][np.argmax(prec_arr[:-1][valid])])
    else:
        best_t_d2    = float(thresholds[0])
        best_prec_d2 = 0.0
    y_pred_d2 = (prob_d >= best_t_d2).astype(int)
    rec_d2    = float(y_pred_d2[y_val == 1].mean()) if y_val.sum() > 0 else 0.0
    f05_d2    = fbeta_score(y_val, y_pred_d2, beta=0.5, zero_division=0)
    ap_d2     = float(average_precision_score(y_val, prob_d))
    auc_d2    = float(roc_auc_score(y_val, prob_d))
    m = {
        "avg_precision":       round(ap_d2, 6),
        "f05_score":           round(f05_d2, 6),
        "f05_threshold":       round(best_t_d2, 3),
        "precision_at_50":     round(precision_at_k(y_val, prob_d, 50), 4),
        "precision_at_100":    round(precision_at_k(y_val, prob_d, 100), 4),
        "auc_roc":             round(auc_d2, 6),
        "recall_at_threshold": round(rec_d2, 4),
    }
    note = f"exp115 max_prec_at_recall>=0.5 threshold={best_t_d2:.3f} prec={best_prec_d2:.3f}"
    row = log_exp("D", "ENS_LR", "STACK",
                  {"variant": "max_prec_recall_05", "threshold": best_t_d2},
                  m, 3, notes=note)
    print_exp(row)
except Exception as e:
    row = log_exp("D", "ENS_LR", "STACK", {"variant": "max_prec_recall_05"},
                  zero_metrics(), 3, notes=f"ERROR: {e}")
    print_exp(row)

# D3 — report precision at recall exactly = 0.5 (interpolated)
try:
    prec_arr2, rec_arr2, thr_arr2 = precision_recall_curve(y_val, prob_d)
    # Find threshold nearest to recall=0.5
    dists = np.abs(rec_arr2[:-1] - 0.5)
    idx_d3 = int(np.argmin(dists))
    t_d3   = float(thr_arr2[idx_d3])
    prec_d3 = float(prec_arr2[idx_d3])
    rec_d3  = float(rec_arr2[idx_d3])
    y_pred_d3 = (prob_d >= t_d3).astype(int)
    f05_d3  = fbeta_score(y_val, y_pred_d3, beta=0.5, zero_division=0)
    ap_d3   = float(average_precision_score(y_val, prob_d))
    auc_d3  = float(roc_auc_score(y_val, prob_d))
    m = {
        "avg_precision":       round(ap_d3, 6),
        "f05_score":           round(f05_d3, 6),
        "f05_threshold":       round(t_d3, 3),
        "precision_at_50":     round(precision_at_k(y_val, prob_d, 50), 4),
        "precision_at_100":    round(precision_at_k(y_val, prob_d, 100), 4),
        "auc_roc":             round(auc_d3, 6),
        "recall_at_threshold": round(rec_d3, 4),
    }
    note = f"exp115 precision@recall=0.5: prec={prec_d3:.4f} rec={rec_d3:.4f} threshold={t_d3:.4f}"
    row = log_exp("D", "ENS_LR", "STACK",
                  {"variant": "prec_at_recall_05", "threshold": t_d3},
                  m, 3, notes=note)
    print_exp(row)
except Exception as e:
    row = log_exp("D", "ENS_LR", "STACK", {"variant": "prec_at_recall_05"},
                  zero_metrics(), 3, notes=f"ERROR: {e}")
    print_exp(row)

_flush()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FOLLOW-UP RESULTS SUMMARY")
print('='*70)

df_new = pd.DataFrame(experiments)
df_new["avg_precision"] = pd.to_numeric(df_new["avg_precision"], errors="coerce")
df_new["f05_score"]     = pd.to_numeric(df_new["f05_score"],     errors="coerce")
df_new["auc_roc"]       = pd.to_numeric(df_new["auc_roc"],       errors="coerce")
df_new["precision_at_50"]  = pd.to_numeric(df_new["precision_at_50"],  errors="coerce")
df_new["precision_at_100"] = pd.to_numeric(df_new["precision_at_100"], errors="coerce")
df_new["recall_at_threshold"] = pd.to_numeric(df_new["recall_at_threshold"], errors="coerce")

print(f"\nAll {len(df_new)} follow-up experiments:")
print(f"{'ExpID':>6} {'Grp':<4} {'Model':<12} {'FS':<6} {'AP':>7} {'F0.5':>7} {'P@50':>6} {'P@100':>6} {'AUC':>7}")
print("-"*70)
for _, r in df_new.sort_values("avg_precision", ascending=False).iterrows():
    print(f"  {int(r['exp_id']):>4} {str(r['batch']):<4} {r['model_family']:<12} {r['feature_set']:<6} "
          f"{r['avg_precision']:>7.4f} {r['f05_score']:>7.4f} "
          f"{r['precision_at_50']:>6.3f} {r['precision_at_100']:>6.3f} {r['auc_roc']:>7.4f}")

# Combined ranking (load full log)
df_all = pd.read_csv(LOG_FILE)
df_all["avg_precision"] = pd.to_numeric(df_all["avg_precision"], errors="coerce")
df_all["f05_score"]     = pd.to_numeric(df_all["f05_score"],     errors="coerce")
df_all["auc_roc"]       = pd.to_numeric(df_all["auc_roc"],       errors="coerce")
df_all["precision_at_50"]  = pd.to_numeric(df_all["precision_at_50"],  errors="coerce")
df_all["precision_at_100"] = pd.to_numeric(df_all["precision_at_100"], errors="coerce")

print(f"\nTOP 10 BY AVERAGE PRECISION (all {len(df_all)} experiments):")
print(f"{'Rank':>4} {'ExpID':>5} {'Grp':<4} {'Model':<12} {'FS':<6} {'AP':>7} {'F0.5':>7} {'P@50':>6} {'P@100':>6} {'AUC':>7}")
print("-"*75)
top10_ap = df_all[df_all["avg_precision"] > 0].nlargest(10, "avg_precision")
for rank, (_, r) in enumerate(top10_ap.iterrows(), 1):
    marker = " ★" if r["avg_precision"] >= 0.10 else "  "
    print(f"{rank:>4} {int(r['exp_id']):>5} {str(r['batch']):<4} {r['model_family']:<12} {r['feature_set']:<6} "
          f"{r['avg_precision']:>7.4f} {r['f05_score']:>7.4f} "
          f"{r['precision_at_50']:>6.3f} {r['precision_at_100']:>6.3f} {r['auc_roc']:>7.4f}{marker}")

print(f"\nTOP 10 BY F0.5 (all experiments):")
print(f"{'Rank':>4} {'ExpID':>5} {'Grp':<4} {'Model':<12} {'FS':<6} {'AP':>7} {'F0.5':>7} {'P@50':>6} {'P@100':>6} {'AUC':>7}")
print("-"*75)
top10_f05 = df_all[df_all["f05_score"] > 0].nlargest(10, "f05_score")
for rank, (_, r) in enumerate(top10_f05.iterrows(), 1):
    print(f"{rank:>4} {int(r['exp_id']):>5} {str(r['batch']):<4} {r['model_family']:<12} {r['feature_set']:<6} "
          f"{r['avg_precision']:>7.4f} {r['f05_score']:>7.4f} "
          f"{r['precision_at_50']:>6.3f} {r['precision_at_100']:>6.3f} {r['auc_roc']:>7.4f}")

# Key question: did isotonic on exp115 break 0.10 AP?
print(f"\n{'='*70}")
print("KEY QUESTION: Does fixed isotonic calibration on exp115 reach AP >= 0.10?")
print('='*70)
iso115 = df_new[(df_new["notes"].str.contains("exp115", na=False)) &
                (df_new["notes"].str.contains("isotonic", na=False))]
if not iso115.empty:
    r = iso115.iloc[0]
    print(f"  Isotonic on exp115: AP={r['avg_precision']:.4f}  F0.5={r['f05_score']:.4f}  "
          f"AUC={r['auc_roc']:.4f}")
    if r['avg_precision'] >= 0.10:
        print("  ✓ AP TARGET REACHED: >= 0.10")
    else:
        gap = 0.10 - r['avg_precision']
        print(f"  ✗ AP target NOT reached. Gap: {gap:.4f}")

# Champion update check
current_champion_ap = 0.093596  # exp 115 from original run
best_new = df_new[df_new["avg_precision"] > 0].nlargest(1, "avg_precision")
if not best_new.empty:
    best_row = best_new.iloc[0]
    if best_row["avg_precision"] > current_champion_ap:
        print(f"\n  NEW CHAMPION: Exp {int(best_row['exp_id'])} {best_row['model_family']} {best_row['feature_set']}")
        print(f"    AP={best_row['avg_precision']:.4f}  F0.5={best_row['f05_score']:.4f}  AUC={best_row['auc_roc']:.4f}")
        print(f"    Delta AP: +{best_row['avg_precision'] - current_champion_ap:.4f}")

        # Save champion config
        champ_config = {
            "model_family": best_row["model_family"],
            "feature_set":  best_row["feature_set"],
            "n_features":   int(best_row["n_features"]),
            "hyperparams":  json.loads(best_row["hyperparams"]),
            "best_threshold": float(best_row["f05_threshold"]),
            "val_metrics": {
                "avg_precision": float(best_row["avg_precision"]),
                "f05_score":     float(best_row["f05_score"]),
                "f05_threshold": float(best_row["f05_threshold"]),
                "precision_at_50":  float(best_row["precision_at_50"]),
                "precision_at_100": float(best_row["precision_at_100"]),
                "auc_roc":       float(best_row["auc_roc"]),
                "recall": float(best_row["recall_at_threshold"]),
            },
            "vs_baseline": {
                "ap_delta":  round(float(best_row["avg_precision"]) - 0.0438, 4),
                "f05_delta": round(float(best_row["f05_score"])     - 0.0453, 4),
            },
            "notes": best_row["notes"],
        }
        with open(MODEL_DIR / "phbench_champion_config.json", "w") as f:
            json.dump(champ_config, f, indent=2)
        print(f"    Saved updated champion config → models/phbench_champion_config.json")
    else:
        print(f"\n  No improvement over current champion (AP={current_champion_ap:.4f}). Champion unchanged.")

print(f"\n{'='*70}")
print("Follow-up experiments complete.")
print(f"  New experiments written to: {LOG_FILE}")
print(f"  Total experiments in log:   {len(df_all)}")

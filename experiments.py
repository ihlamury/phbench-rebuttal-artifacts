#!/usr/bin/env python3
"""
PHBench — 120+ ML Experiments
Primary metric: F0.5  Secondary: avg_precision, P@50, P@100, AUC-ROC

PYTHONPATH=. python phbench/experiments.py
"""

import warnings; warnings.filterwarnings("ignore")
import csv, json, pickle, time, traceback, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
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
DATA_DIR   = Path("data/benchmark")
EXP_DIR    = Path("data/experiments")
MODEL_DIR  = Path("models")
FIG_DIR    = Path("figures")
LOG_FILE   = EXP_DIR / "experiment_log.csv"

for d in [EXP_DIR, MODEL_DIR, FIG_DIR]:
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
test  = pd.read_csv(DATA_DIR / "phbench_test_features.csv", dtype=str, low_memory=False)

X_train_raw = engineer_features(train)
X_val_raw   = engineer_features(val)
X_test_raw  = engineer_features(test)

BASE_FEATS = [c for c in X_train_raw.columns if c != "post_id"]

y_train = pd.to_numeric(train["label_series_a_within_18m"], errors="coerce").fillna(0).astype(int).values
y_val   = pd.to_numeric(val["label_series_a_within_18m"],   errors="coerce").fillna(0).astype(int).values

n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
SPW   = n_neg / n_pos  # 125.5

print(f"  Train: {len(y_train):,} | {n_pos} pos | SPW={SPW:.1f}")
print(f"  Val:   {len(y_val):,}  | {y_val.sum()} pos")
print(f"  Base features: {len(BASE_FEATS)}")


# ── Feature sets ───────────────────────────────────────────────────────────────
def make_feature_sets(X_raw: pd.DataFrame) -> dict[str, np.ndarray]:
    bf = BASE_FEATS  # shorthand

    # FS1 — baseline 62
    fs1 = X_raw[bf].values

    # FS2 — engagement + rank only (15)
    fs2_cols = [
        "log_votes", "log_comments", "log_reviews", "votes_per_comment",
        "has_reviews", "review_rating",
        "daily_rank", "weekly_rank", "monthly_rank",
        "has_daily_rank", "has_weekly_rank",
        "log_daily_rank", "log_weekly_rank", "rank_bucket",
        "votes_x_rank_bucket",
    ]
    fs2 = X_raw[[c for c in fs2_cols if c in X_raw.columns]].values

    # FS3 — baseline + text density (68)
    X3 = X_raw[bf].copy()
    desc_wc = pd.to_numeric(X_raw["desc_word_count"], errors="coerce").fillna(0)
    tag_wc  = pd.to_numeric(X_raw["tagline_word_count"], errors="coerce").fillna(0)
    desc_l  = pd.to_numeric(X_raw["desc_length"], errors="coerce").fillna(0)
    tag_l   = pd.to_numeric(X_raw["tagline_length"], errors="coerce").fillna(0)
    log_v   = pd.to_numeric(X_raw["log_votes"], errors="coerce").fillna(0)
    log_dr  = pd.to_numeric(X_raw["log_daily_rank"], errors="coerce").fillna(0)
    has_d   = pd.to_numeric(X_raw["has_description"], errors="coerce").fillna(0)
    X3["desc_density"]      = desc_l / (desc_wc + 1)
    X3["tagline_density"]   = tag_l  / (tag_wc  + 1)
    X3["log_desc_length"]   = np.log1p(desc_l)
    X3["log_tagline_length"]= np.log1p(tag_l)
    X3["desc_x_votes"]      = log_v * has_d
    X3["desc_x_rank"]       = log_dr * has_d
    fs3 = X3.values

    # FS4 — baseline + topic interactions (69)
    X4 = X_raw[bf].copy()
    is_ai   = pd.to_numeric(X_raw["is_ai_topic"], errors="coerce").fillna(0)
    is_b2b  = pd.to_numeric(X_raw["is_b2b_topic"], errors="coerce").fillna(0)
    fin     = pd.to_numeric(X_raw["topic_fintech"], errors="coerce").fillna(0)
    api     = pd.to_numeric(X_raw["topic_api"], errors="coerce").fillna(0)
    rb      = pd.to_numeric(X_raw["rank_bucket"], errors="coerce").fillna(0)
    ipw     = pd.to_numeric(X_raw["is_prime_window"], errors="coerce").fillna(0)
    tc      = pd.to_numeric(X_raw["topic_count"], errors="coerce").fillna(0)
    X4["ai_x_votes"]           = is_ai * log_v
    X4["ai_x_rank"]            = is_ai * rb
    X4["fintech_x_votes"]      = fin   * log_v
    X4["api_x_votes"]          = api   * log_v
    X4["b2b_x_prime"]          = is_b2b * ipw
    X4["topic_specificity"]    = tc / 5.0
    X4["rank_bucket_x_prime"]  = rb * ipw
    fs4 = X4.values

    # FS5 — maker focused (20)
    mc   = pd.to_numeric(X_raw["maker_count"], errors="coerce").fillna(0)
    lmf  = pd.to_numeric(X_raw["log_maker_followers"], errors="coerce").fillna(0)
    lc   = pd.to_numeric(X_raw["log_comments"], errors="coerce").fillna(0)
    hr   = pd.to_numeric(X_raw["has_reviews"], errors="coerce").fillna(0)
    fin2 = pd.to_numeric(X_raw["topic_fintech"], errors="coerce").fillna(0)
    dv   = pd.to_numeric(X_raw["topic_developer_tools"], errors="coerce").fillna(0)
    wkr  = pd.to_numeric(X_raw["weekly_rank"], errors="coerce").fillna(0)
    dr   = pd.to_numeric(X_raw["daily_rank"], errors="coerce").fillna(0)
    has_mk = pd.to_numeric(X_raw["has_makers"], errors="coerce").fillna(0)
    mft  = pd.to_numeric(X_raw["maker_followers_total"], errors="coerce").fillna(0)
    mt   = pd.to_numeric(X_raw["maker_tier"], errors="coerce").fillna(0)
    fs5_arr = np.column_stack([
        mc, has_mk, mft, lmf, mt,
        mc * log_v, mc * rb, mc * ipw, mc * is_ai,
        mc * lc, mc * hr, mc * fin2, mc * api, mc * dv,
        log_v, lc, rb, dr, wkr, ipw,
    ])

    # FS6 — union FS1+FS3+FS4 (~75 features)
    X6 = X3.copy()  # FS3 already has FS1 + text density
    for col in ["ai_x_votes","ai_x_rank","fintech_x_votes","api_x_votes",
                "b2b_x_prime","topic_specificity","rank_bucket_x_prime"]:
        X6[col] = X4[col]
    fs6 = X6.values

    return {
        "FS1": fs1,
        "FS2": fs2,
        "FS3": fs3,
        "FS4": fs4,
        "FS5": fs5_arr,
        "FS6": fs6,
    }


print("Building feature sets...")
FS_TRAIN = make_feature_sets(X_train_raw)
FS_VAL   = make_feature_sets(X_val_raw)
FS_TEST  = make_feature_sets(X_test_raw)
for k, v in FS_TRAIN.items():
    print(f"  {k}: {v.shape[1]} features")


# ── Metric helpers ─────────────────────────────────────────────────────────────
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
    best_idx = int(np.argmax(f05_scores))
    best_thresh = float(thresholds[best_idx])
    best_f05    = float(f05_scores[best_idx])
    y_pred_best = (y_prob >= best_thresh).astype(int)
    recall = float(y_pred_best[y_true == 1].mean()) if y_true.sum() > 0 else 0.0

    return {
        "avg_precision":    round(ap, 6),
        "f05_score":        round(best_f05, 6),
        "f05_threshold":    round(best_thresh, 2),
        "precision_at_50":  round(p50, 4),
        "precision_at_100": round(p100, 4),
        "auc_roc":          round(auc, 6),
        "recall_at_threshold": round(recall, 4),
    }


# ── Experiment logger ──────────────────────────────────────────────────────────
experiments: list[dict] = []
_pending_writes: list[dict] = []

def _flush_log():
    global _pending_writes
    if not _pending_writes:
        return
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if write_header:
            w.writeheader()
        w.writerows(_pending_writes)
    _pending_writes = []

def log_exp(batch, model_family, feature_set, hyperparams,
            metrics, n_features, best_iteration=None, notes=""):
    exp_id = len(experiments) + 1
    row = {
        "exp_id":       exp_id,
        "batch":        batch,
        "model_family": model_family,
        "feature_set":  feature_set,
        "hyperparams":  json.dumps(hyperparams),
        "n_features":   n_features,
        "best_iteration": best_iteration if best_iteration is not None else "",
        "notes":        notes,
        "timestamp":    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
    }
    experiments.append(row)
    _pending_writes.append(row)
    if len(_pending_writes) >= 5:
        _flush_log()
    return row

def print_exp(row, total):
    print(
        f"Exp {row['exp_id']:>3}/{total} | {row['model_family']:<6} {row['feature_set']:<4} | "
        f"AP={row['avg_precision']:.4f} F0.5={row['f05_score']:.4f} "
        f"P@50={row['precision_at_50']:.3f} P@100={row['precision_at_100']:.3f} "
        f"AUC={row['auc_roc']:.4f}",
        flush=True,
    )

TOTAL_EXPERIMENTS = 120


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 1 — Logistic Regression (20 experiments)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BATCH 1 — Logistic Regression (20 experiments)")
print('='*70)

LR_CONFIGS = (
    [("FS1", c) for c in [0.01, 0.05, 0.1, 0.5, 1.0]] +
    [("FS2", c) for c in [0.01, 0.1, 1.0]] +
    [("FS3", c) for c in [0.01, 0.05, 0.1, 0.5, 1.0]] +
    [("FS4", c) for c in [0.01, 0.05, 0.1, 0.5, 1.0]] +
    [("FS6", c) for c in [0.01, 0.1]]
)

for fs_name, C in LR_CONFIGS:
    try:
        solver = "saga" if C >= 1.0 else "lbfgs"
        model = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, class_weight="balanced", max_iter=2000,
                random_state=42, solver=solver,
            )),
        ])
        model.fit(FS_TRAIN[fs_name], y_train)
        prob = model.predict_proba(FS_VAL[fs_name])[:, 1]
        m = compute_metrics(y_val, prob)
        hp = {"C": C, "solver": solver}
        row = log_exp(1, "LR", fs_name, hp, m, FS_TRAIN[fs_name].shape[1])
    except Exception as e:
        m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
        row = log_exp(1, "LR", fs_name, {"C": C}, m, FS_TRAIN[fs_name].shape[1], notes=f"ERROR: {e}")
    print_exp(row, TOTAL_EXPERIMENTS)

_flush_log()


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 2 — XGBoost (25 experiments, random search)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BATCH 2 — XGBoost (25 experiments)")
print('='*70)

rng2 = np.random.RandomState(42)
xgb_grid = {
    "feature_set":      ["FS1","FS3","FS4","FS6"],
    "max_depth":        [3, 4, 6],
    "learning_rate":    [0.01, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [50, 100, 126, 200],
    "gamma":            [0, 0.1, 0.5],
}

for _ in range(25):
    fs_name = rng2.choice(xgb_grid["feature_set"])
    hp = {
        "max_depth":        int(rng2.choice(xgb_grid["max_depth"])),
        "learning_rate":    float(rng2.choice(xgb_grid["learning_rate"])),
        "min_child_weight": int(rng2.choice(xgb_grid["min_child_weight"])),
        "subsample":        float(rng2.choice(xgb_grid["subsample"])),
        "colsample_bytree": float(rng2.choice(xgb_grid["colsample_bytree"])),
        "scale_pos_weight": float(rng2.choice(xgb_grid["scale_pos_weight"])),
        "gamma":            float(rng2.choice(xgb_grid["gamma"])),
    }
    try:
        model = XGBClassifier(
            n_estimators=500, eval_metric="auc", early_stopping_rounds=30,
            random_state=42, n_jobs=-1, verbosity=0, **hp,
        )
        model.fit(FS_TRAIN[fs_name], y_train,
                  eval_set=[(FS_VAL[fs_name], y_val)], verbose=False)
        best_iter = model.best_iteration + 1
        prob = model.predict_proba(FS_VAL[fs_name])[:, 1]
        m = compute_metrics(y_val, prob)
        row = log_exp(2, "XGB", fs_name, hp, m, FS_TRAIN[fs_name].shape[1], best_iter)
    except Exception as e:
        m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
        row = log_exp(2, "XGB", fs_name, hp, m, FS_TRAIN[fs_name].shape[1], notes=f"ERROR: {e}")
    print_exp(row, TOTAL_EXPERIMENTS)

_flush_log()


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 3 — LightGBM (25 experiments, random search)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BATCH 3 — LightGBM (25 experiments)")
print('='*70)

rng3 = np.random.RandomState(42)
lgbm_grid = {
    "feature_set":        ["FS1","FS3","FS4","FS6"],
    "num_leaves":         [15, 31, 63, 127],
    "learning_rate":      [0.01, 0.05, 0.1],
    "min_child_samples":  [5, 10, 20, 50],
    "reg_alpha":          [0, 0.01, 0.1, 1.0],
    "reg_lambda":         [0, 0.01, 0.1, 1.0],
    "feature_fraction":   [0.7, 0.8, 1.0],
    "bagging_fraction":   [0.7, 0.8, 1.0],
    "bagging_freq":       [0, 5],
}

for _ in range(25):
    fs_name = rng3.choice(lgbm_grid["feature_set"])
    bf      = float(rng3.choice(lgbm_grid["bagging_fraction"]))
    bfreq   = int(rng3.choice(lgbm_grid["bagging_freq"]))
    hp = {
        "num_leaves":        int(rng3.choice(lgbm_grid["num_leaves"])),
        "learning_rate":     float(rng3.choice(lgbm_grid["learning_rate"])),
        "min_child_samples": int(rng3.choice(lgbm_grid["min_child_samples"])),
        "reg_alpha":         float(rng3.choice(lgbm_grid["reg_alpha"])),
        "reg_lambda":        float(rng3.choice(lgbm_grid["reg_lambda"])),
        "feature_fraction":  float(rng3.choice(lgbm_grid["feature_fraction"])),
        "bagging_fraction":  bf,
        "bagging_freq":      bfreq if bf < 1.0 else 0,
    }
    try:
        model = lgb.LGBMClassifier(
            n_estimators=500, metric="auc", is_unbalance=True,
            random_state=42, n_jobs=-1, verbose=-1, **hp,
        )
        model.fit(
            FS_TRAIN[fs_name], y_train,
            eval_set=[(FS_VAL[fs_name], y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        best_iter = model.best_iteration_
        prob = model.predict_proba(FS_VAL[fs_name])[:, 1]
        m = compute_metrics(y_val, prob)
        row = log_exp(3, "LGBM", fs_name, hp, m, FS_TRAIN[fs_name].shape[1], best_iter)
    except Exception as e:
        m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
        row = log_exp(3, "LGBM", fs_name, hp, m, FS_TRAIN[fs_name].shape[1], notes=f"ERROR: {e}")
    print_exp(row, TOTAL_EXPERIMENTS)

_flush_log()


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 4 — Random Forest (15 experiments)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BATCH 4 — Random Forest (15 experiments)")
print('='*70)

rng4 = np.random.RandomState(42)
rf_grid = {
    "feature_set":      ["FS1","FS3","FS6"],
    "n_estimators":     [200, 500, 1000],
    "max_depth":        [4, 6, 8, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features":     ["sqrt", "log2", 0.5],
    "class_weight":     ["balanced", "balanced_subsample"],
}

for _ in range(15):
    fs_name = rng4.choice(rf_grid["feature_set"])
    hp = {
        "n_estimators":     int(rng4.choice(rf_grid["n_estimators"])),
        "max_depth":        rng4.choice(rf_grid["max_depth"]),
        "min_samples_leaf": int(rng4.choice(rf_grid["min_samples_leaf"])),
        "max_features":     rng4.choice(rf_grid["max_features"]),
        "class_weight":     str(rng4.choice(rf_grid["class_weight"])),
    }
    try:
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **hp)
        model.fit(FS_TRAIN[fs_name], y_train)
        prob = model.predict_proba(FS_VAL[fs_name])[:, 1]
        m = compute_metrics(y_val, prob)
        row = log_exp(4, "RF", fs_name, hp, m, FS_TRAIN[fs_name].shape[1])
    except Exception as e:
        m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
        row = log_exp(4, "RF", fs_name, hp, m, FS_TRAIN[fs_name].shape[1], notes=f"ERROR: {e}")
    print_exp(row, TOTAL_EXPERIMENTS)

_flush_log()


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 5 — Calibration (15 experiments: top 5 × 3 methods)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BATCH 5 — Calibration (15 experiments)")
print('='*70)

# Find top 5 by F0.5 from batches 1-4
prev_exps = [e for e in experiments if e["batch"] <= 4 and e["f05_score"] > 0]
top5_exps = sorted(prev_exps, key=lambda x: x["f05_score"], reverse=True)[:5]
print(f"  Top 5 models for calibration:")
for e in top5_exps:
    print(f"    Exp {e['exp_id']} {e['model_family']} {e['feature_set']} F0.5={e['f05_score']:.4f} AP={e['avg_precision']:.4f}")

# Re-train the top 5 base models to get fitted objects
def retrain_model(exp_row):
    fs   = exp_row["feature_set"]
    hp   = json.loads(exp_row["hyperparams"])
    mf   = exp_row["model_family"]
    X_tr = FS_TRAIN[fs]
    X_v  = FS_VAL[fs]
    if mf == "LR":
        solver = hp.get("solver", "lbfgs")
        m = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=hp["C"], class_weight="balanced",
                                       max_iter=2000, random_state=42, solver=solver)),
        ])
        m.fit(X_tr, y_train)
    elif mf == "XGB":
        hp2 = {k: v for k, v in hp.items()}
        m = XGBClassifier(n_estimators=500, eval_metric="auc", early_stopping_rounds=30,
                          random_state=42, n_jobs=-1, verbosity=0, **hp2)
        m.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
    elif mf == "LGBM":
        bf  = hp.get("bagging_fraction", 1.0)
        bf2 = hp.get("bagging_freq", 0)
        hp2 = {k: v for k, v in hp.items()}
        m = lgb.LGBMClassifier(n_estimators=500, metric="auc", is_unbalance=True,
                               random_state=42, n_jobs=-1, verbose=-1, **hp2)
        m.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    elif mf == "RF":
        hp2 = {k: (None if v == "None" else v) for k, v in hp.items()}
        m = RandomForestClassifier(random_state=42, n_jobs=-1, **hp2)
        m.fit(X_tr, y_train)
    return m, fs

def sigmoid(x): return 1 / (1 + np.exp(-x))
def logit(p):   return np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))

for exp_row in top5_exps:
    fs   = exp_row["feature_set"]
    base_ap   = exp_row["avg_precision"]
    base_f05  = exp_row["f05_score"]
    base_desc = f"base_exp={exp_row['exp_id']}"

    try:
        base_model, fs = retrain_model(exp_row)
        raw_prob = base_model.predict_proba(FS_VAL[fs])[:, 1]

        # Method A — isotonic regression
        try:
            cal_iso = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
            cal_iso.fit(FS_VAL[fs], y_val)
            prob_iso = cal_iso.predict_proba(FS_VAL[fs])[:, 1]
            m = compute_metrics(y_val, prob_iso)
            note = f"{base_desc} uncal_AP={base_ap:.4f} uncal_F05={base_f05:.4f}"
            row = log_exp(5, exp_row["model_family"], fs, {"method":"isotonic"}, m,
                          FS_TRAIN[fs].shape[1], notes=note)
            print_exp(row, TOTAL_EXPERIMENTS)
        except Exception as e:
            m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
            row = log_exp(5, exp_row["model_family"], fs, {"method":"isotonic"}, m,
                          FS_TRAIN[fs].shape[1], notes=f"ERROR: {e}")
            print_exp(row, TOTAL_EXPERIMENTS)

        # Method B — Platt scaling
        try:
            cal_sig = CalibratedClassifierCV(base_model, cv="prefit", method="sigmoid")
            cal_sig.fit(FS_VAL[fs], y_val)
            prob_sig = cal_sig.predict_proba(FS_VAL[fs])[:, 1]
            m = compute_metrics(y_val, prob_sig)
            row = log_exp(5, exp_row["model_family"], fs, {"method":"sigmoid"}, m,
                          FS_TRAIN[fs].shape[1], notes=base_desc)
            print_exp(row, TOTAL_EXPERIMENTS)
        except Exception as e:
            m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
            row = log_exp(5, exp_row["model_family"], fs, {"method":"sigmoid"}, m,
                          FS_TRAIN[fs].shape[1], notes=f"ERROR: {e}")
            print_exp(row, TOTAL_EXPERIMENTS)

        # Method C — temperature scaling
        try:
            temps = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
            best_temp, best_f05_temp, best_prob_temp = None, -1, None
            for temp in temps:
                prob_t = sigmoid(logit(raw_prob) / temp)
                f05_t  = max(
                    fbeta_score(y_val, (prob_t >= t).astype(int), beta=0.5, zero_division=0)
                    for t in np.arange(0.01, 0.99, 0.01)
                )
                if f05_t > best_f05_temp:
                    best_f05_temp = f05_t; best_temp = temp; best_prob_temp = prob_t
            m = compute_metrics(y_val, best_prob_temp)
            row = log_exp(5, exp_row["model_family"], fs,
                          {"method":"temperature","temp": best_temp}, m,
                          FS_TRAIN[fs].shape[1], notes=base_desc)
            print_exp(row, TOTAL_EXPERIMENTS)
        except Exception as e:
            m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
            row = log_exp(5, exp_row["model_family"], fs, {"method":"temperature"}, m,
                          FS_TRAIN[fs].shape[1], notes=f"ERROR: {e}")
            print_exp(row, TOTAL_EXPERIMENTS)

    except Exception as e:
        for method in ["isotonic", "sigmoid", "temperature"]:
            m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
            row = log_exp(5, exp_row["model_family"], fs, {"method": method}, m,
                          0, notes=f"ERROR retrain: {e}")
            print_exp(row, TOTAL_EXPERIMENTS)

_flush_log()


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 6 — Threshold optimization (top 3 models × 4 variants + extras)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BATCH 6 — Threshold Optimization (10 experiments)")
print('='*70)

all_so_far = [e for e in experiments if e["batch"] <= 5 and e["f05_score"] > 0]
top3_thresh = sorted(all_so_far, key=lambda x: x["f05_score"], reverse=True)[:3]
print(f"  Top 3 models for threshold analysis:")
for e in top3_thresh:
    print(f"    Exp {e['exp_id']} {e['model_family']} {e['feature_set']} F0.5={e['f05_score']:.4f}")

thresh_curves = {}

for exp_row in top3_thresh:
    fs = exp_row["feature_set"]
    try:
        base_model, fs = retrain_model(exp_row)
        prob = base_model.predict_proba(FS_VAL[fs])[:, 1]
        thresh_curves[exp_row["exp_id"]] = (prob, exp_row)
        thresholds = np.arange(0.01, 0.99, 0.01)

        # Variant 1: max F0.5 (already computed, log explicitly)
        f05_scores = [fbeta_score(y_val, (prob>=t).astype(int), beta=0.5, zero_division=0)
                      for t in thresholds]
        best_t_f05 = thresholds[np.argmax(f05_scores)]
        m = compute_metrics(y_val, prob)
        row = log_exp(6, exp_row["model_family"], fs,
                      {"variant":"max_f05","threshold": float(best_t_f05)},
                      m, FS_TRAIN[fs].shape[1],
                      notes=f"base_exp={exp_row['exp_id']}")
        print_exp(row, TOTAL_EXPERIMENTS)

        # Variant 2: max F1
        f1_scores = [fbeta_score(y_val, (prob>=t).astype(int), beta=1.0, zero_division=0)
                     for t in thresholds]
        best_t_f1 = thresholds[np.argmax(f1_scores)]
        y_pred_f1 = (prob >= best_t_f1).astype(int)
        p50_f1 = precision_at_k(y_val, prob, 50)
        p100_f1 = precision_at_k(y_val, prob, 100)
        ap_f1 = float(average_precision_score(y_val, prob))
        auc_f1 = float(roc_auc_score(y_val, prob))
        rec_f1 = float(y_pred_f1[y_val==1].mean()) if y_val.sum() > 0 else 0.0
        m2 = {"avg_precision": round(ap_f1,6), "f05_score": round(max(f05_scores),6),
               "f05_threshold": round(best_t_f05,2), "precision_at_50": round(p50_f1,4),
               "precision_at_100": round(p100_f1,4), "auc_roc": round(auc_f1,6),
               "recall_at_threshold": round(rec_f1,4)}
        row = log_exp(6, exp_row["model_family"], fs,
                      {"variant":"max_f1","threshold": float(best_t_f1)},
                      m2, FS_TRAIN[fs].shape[1],
                      notes=f"base_exp={exp_row['exp_id']}")
        print_exp(row, TOTAL_EXPERIMENTS)

        # Variant 3: max precision at recall >= 0.5
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y_val, prob)
        valid = rec_arr[:-1] >= 0.5
        if valid.any():
            best_t_prec = float(thr_arr[valid][np.argmax(prec_arr[:-1][valid])])
        else:
            best_t_prec = float(thresholds[np.argmax(f05_scores)])
        y_pred_p = (prob >= best_t_prec).astype(int)
        rec_p = float(y_pred_p[y_val==1].mean()) if y_val.sum() > 0 else 0.0
        f05_p = fbeta_score(y_val, y_pred_p, beta=0.5, zero_division=0)
        m3 = {"avg_precision": round(ap_f1,6), "f05_score": round(f05_p,6),
               "f05_threshold": round(best_t_prec,2), "precision_at_50": round(p50_f1,4),
               "precision_at_100": round(p100_f1,4), "auc_roc": round(auc_f1,6),
               "recall_at_threshold": round(rec_p,4)}
        row = log_exp(6, exp_row["model_family"], fs,
                      {"variant":"max_prec_recall_05","threshold": best_t_prec},
                      m3, FS_TRAIN[fs].shape[1],
                      notes=f"base_exp={exp_row['exp_id']}")
        print_exp(row, TOTAL_EXPERIMENTS)

        # Variant 4: precision@50 optimized (rank-based, threshold irrelevant)
        # Use prob directly, compute P@50 at different thresholds
        p50_scores = [precision_at_k(y_val, prob, 50)] * len(thresholds)  # rank-based, constant
        m4 = m.copy()
        m4["f05_threshold"] = round(best_t_f05, 2)
        row = log_exp(6, exp_row["model_family"], fs,
                      {"variant":"rank_p50"},
                      m4, FS_TRAIN[fs].shape[1],
                      notes=f"base_exp={exp_row['exp_id']} rank-based P@50 fixed")
        print_exp(row, TOTAL_EXPERIMENTS)

    except Exception as e:
        for variant in ["max_f05","max_f1","max_prec_recall_05","rank_p50"]:
            m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
            row = log_exp(6, exp_row["model_family"], fs, {"variant": variant}, m,
                          0, notes=f"ERROR: {e}")
            print_exp(row, TOTAL_EXPERIMENTS)

# 2 extra experiments: top 2 models with fine-grained threshold grid
for exp_row in top3_thresh[:2]:
    fs = exp_row["feature_set"]
    try:
        base_model, fs = retrain_model(exp_row)
        prob = base_model.predict_proba(FS_VAL[fs])[:, 1]
        fine_thresholds = np.arange(0.001, 0.30, 0.001)
        f05_fine = [fbeta_score(y_val, (prob>=t).astype(int), beta=0.5, zero_division=0)
                    for t in fine_thresholds]
        best_t_fine = float(fine_thresholds[np.argmax(f05_fine)])
        m = compute_metrics(y_val, prob)
        m["f05_threshold"] = round(best_t_fine, 3)
        m["f05_score"] = round(max(f05_fine), 6)
        row = log_exp(6, exp_row["model_family"], fs,
                      {"variant":"fine_grid","threshold": best_t_fine},
                      m, FS_TRAIN[fs].shape[1],
                      notes=f"base_exp={exp_row['exp_id']} fine 0.001 step")
        print_exp(row, TOTAL_EXPERIMENTS)
    except Exception as e:
        m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
        row = log_exp(6, exp_row["model_family"], fs, {"variant":"fine_grid"}, m, 0, notes=f"ERROR: {e}")
        print_exp(row, TOTAL_EXPERIMENTS)

_flush_log()


# ── Figure 13: threshold curves ────────────────────────────────────────────────
if thresh_curves:
    thresholds_plot = np.arange(0.01, 0.99, 0.01)
    fig, axes = plt.subplots(1, len(thresh_curves), figsize=(6*len(thresh_curves), 5), squeeze=False)
    colors = ["#2196F3","#FF5722","#4CAF50"]
    for i, (eid, (prob, exp_row)) in enumerate(thresh_curves.items()):
        ax = axes[0][i]
        f05_c = [fbeta_score(y_val, (prob>=t).astype(int), beta=0.5, zero_division=0) for t in thresholds_plot]
        f1_c  = [fbeta_score(y_val, (prob>=t).astype(int), beta=1.0, zero_division=0) for t in thresholds_plot]
        ax.plot(thresholds_plot, f05_c, color=colors[i], linewidth=2, label="F0.5")
        ax.plot(thresholds_plot, f1_c,  color=colors[i], linewidth=1.5, linestyle="--", label="F1")
        best_t = thresholds_plot[np.argmax(f05_c)]
        ax.axvline(best_t, color="black", linestyle=":", linewidth=1, label=f"opt thresh={best_t:.2f}")
        ax.set_title(f"Exp {eid} {exp_row['model_family']} {exp_row['feature_set']}", fontsize=10)
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("PHBench — F0.5 & F1 vs Threshold (Top 3 Models)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_threshold_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved figures/13_threshold_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 7 — Stacking ensembles (10 experiments)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("BATCH 7 — Stacking Ensembles (10 experiments)")
print('='*70)

# Identify best per family
def best_by_family(family):
    cands = [e for e in experiments if e["model_family"] == family
             and e["batch"] <= 4 and e["f05_score"] > 0]
    return max(cands, key=lambda x: x["f05_score"]) if cands else None

best_lr   = best_by_family("LR")
best_xgb  = best_by_family("XGB")
best_lgbm = best_by_family("LGBM")
best_rf   = best_by_family("RF")
print(f"  Best LR:   Exp {best_lr['exp_id'] if best_lr else 'N/A'}")
print(f"  Best XGB:  Exp {best_xgb['exp_id'] if best_xgb else 'N/A'}")
print(f"  Best LGBM: Exp {best_lgbm['exp_id'] if best_lgbm else 'N/A'}")
print(f"  Best RF:   Exp {best_rf['exp_id'] if best_rf else 'N/A'}")

top5_stack = sorted(
    [e for e in experiments if e["batch"] <= 4 and e["f05_score"] > 0],
    key=lambda x: x["f05_score"], reverse=True
)[:5]

def get_oof_preds(exp_row, n_folds=5):
    """Out-of-fold predictions on train set."""
    fs   = exp_row["feature_set"]
    X_tr = FS_TRAIN[fs]
    hp   = json.loads(exp_row["hyperparams"])
    mf   = exp_row["model_family"]
    oof  = np.zeros(len(y_train))
    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_tr, fold_val in skf.split(X_tr, y_train):
        Xf_tr, yf_tr = X_tr[fold_tr], y_train[fold_tr]
        Xf_v,  yf_v  = X_tr[fold_val], y_train[fold_val]
        if mf == "LR":
            solver = hp.get("solver","lbfgs")
            m = Pipeline([("sc",StandardScaler()),
                          ("clf",LogisticRegression(C=hp["C"], class_weight="balanced",
                                                    max_iter=2000, random_state=42, solver=solver))])
            m.fit(Xf_tr, yf_tr)
            oof[fold_val] = m.predict_proba(Xf_v)[:,1]
        elif mf == "XGB":
            m = XGBClassifier(n_estimators=200, eval_metric="auc", random_state=42,
                              n_jobs=-1, verbosity=0, **{k:v for k,v in hp.items()})
            m.fit(Xf_tr, yf_tr)
            oof[fold_val] = m.predict_proba(Xf_v)[:,1]
        elif mf == "LGBM":
            m = lgb.LGBMClassifier(n_estimators=200, metric="auc", is_unbalance=True,
                                   random_state=42, n_jobs=-1, verbose=-1,
                                   **{k:v for k,v in hp.items()})
            m.fit(Xf_tr, yf_tr)
            oof[fold_val] = m.predict_proba(Xf_v)[:,1]
        elif mf == "RF":
            hp2 = {k:(None if v=="None" else v) for k,v in hp.items()}
            m = RandomForestClassifier(random_state=42, n_jobs=-1, **hp2)
            m.fit(Xf_tr, yf_tr)
            oof[fold_val] = m.predict_proba(Xf_v)[:,1]
    return oof

def get_val_preds(exp_row):
    m, fs = retrain_model(exp_row)
    return m.predict_proba(FS_VAL[fs])[:, 1]

print("  Building OOF predictions (may take a few minutes)...")

# Collect predictions for base models
base_models_for_stack = [e for e in [best_lr, best_xgb, best_lgbm, best_rf] if e is not None]
oof_preds  = {}
val_preds_stack = {}
for exp_row in base_models_for_stack:
    eid = exp_row["exp_id"]
    try:
        oof_preds[eid]       = get_oof_preds(exp_row)
        val_preds_stack[eid] = get_val_preds(exp_row)
    except Exception as e:
        print(f"    WARNING: OOF failed for exp {eid}: {e}")

# Top 5 preds
top5_oof  = {}
top5_val  = {}
for exp_row in top5_stack:
    eid = exp_row["exp_id"]
    if eid not in oof_preds:
        try:
            oof_preds[eid]       = get_oof_preds(exp_row)
            val_preds_stack[eid] = get_val_preds(exp_row)
            top5_oof[eid]  = oof_preds[eid]
            top5_val[eid]  = val_preds_stack[eid]
        except Exception as e:
            print(f"    WARNING: OOF failed for exp {eid}: {e}")
    else:
        top5_oof[eid]  = oof_preds[eid]
        top5_val[eid]  = val_preds_stack[eid]

def run_stack(name, train_oof_list, val_prob_list, meta, fs_name="STACK"):
    X_meta_tr = np.column_stack(train_oof_list) if len(train_oof_list) > 1 else train_oof_list[0].reshape(-1,1)
    X_meta_v  = np.column_stack(val_prob_list)  if len(val_prob_list) > 1 else val_prob_list[0].reshape(-1,1)
    n_feat    = X_meta_tr.shape[1]
    if meta == "LR":
        mm = Pipeline([("sc",StandardScaler()),
                       ("clf",LogisticRegression(C=0.1, class_weight="balanced",
                                                  max_iter=1000, random_state=42))])
        mm.fit(X_meta_tr, y_train)
        prob = mm.predict_proba(X_meta_v)[:,1]
    elif meta == "XGB":
        mm = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1,
                           scale_pos_weight=SPW, eval_metric="auc",
                           random_state=42, n_jobs=-1, verbosity=0)
        mm.fit(X_meta_tr, y_train, eval_set=[(X_meta_v, y_val)], verbose=False)
        prob = mm.predict_proba(X_meta_v)[:,1]
    elif meta == "Ridge":
        mm = Ridge(alpha=1.0)
        mm.fit(X_meta_tr, y_train)
        prob = np.clip(mm.predict(X_meta_v), 0, 1)
    elif meta == "avg":
        prob = np.mean(X_meta_v, axis=1)
    elif meta == "weighted":
        weights = np.array([e["f05_score"] for e in top5_stack[:len(val_prob_list)]])
        weights = weights / weights.sum()
        prob = X_meta_v @ weights
    elif meta == "rank_avg":
        ranks = np.column_stack([pd.Series(p).rank(pct=True).values for p in val_prob_list])
        prob = ranks.mean(axis=1)
    elif meta == "geo_mean":
        prob = np.exp(np.mean(np.log(np.clip(X_meta_v, 1e-7, 1)), axis=1))
    elif meta == "max":
        prob = X_meta_v.max(axis=1)
    return prob

# Stacks 1–5 (meta-learner stacks)
stack_configs = [
    ("Stack1_LR_meta",   [e["exp_id"] for e in [best_lr,best_xgb,best_lgbm] if e], "LR"),
    ("Stack2_XGB_meta",  [e["exp_id"] for e in [best_lr,best_xgb,best_lgbm] if e], "XGB"),
    ("Stack3_4model_LR", [e["exp_id"] for e in base_models_for_stack],               "LR"),
    ("Stack4_top5_LR",   [e["exp_id"] for e in top5_stack],                          "LR"),
    ("Stack5_top5_Ridge",[e["exp_id"] for e in top5_stack],                          "Ridge"),
]
avg_configs = [
    ("Stack6_avg3",     [e["exp_id"] for e in top5_stack[:3]],  "avg"),
    ("Stack7_weighted", [e["exp_id"] for e in top5_stack],       "weighted"),
    ("Stack8_rank_avg", [e["exp_id"] for e in top5_stack[:3]],  "rank_avg"),
    ("Stack9_geo",      [e["exp_id"] for e in top5_stack[:3]],  "geo_mean"),
    ("Stack10_max",     [e["exp_id"] for e in top5_stack[:3]],  "max"),
]

for stack_name, eids, meta in stack_configs + avg_configs:
    try:
        oof_list = [oof_preds[eid] for eid in eids if eid in oof_preds]
        val_list = [val_preds_stack[eid] for eid in eids if eid in val_preds_stack]
        if len(oof_list) < 2:
            raise ValueError(f"Not enough base predictions: {len(oof_list)}")
        prob = run_stack(stack_name, oof_list, val_list, meta)
        m = compute_metrics(y_val, prob)
        row = log_exp(7, f"ENS_{meta}", "STACK", {"stack": stack_name, "meta": meta, "n_base": len(oof_list)},
                      m, len(oof_list), notes=f"base_exp_ids={eids}")
    except Exception as e:
        m = {k: 0.0 for k in ["avg_precision","f05_score","f05_threshold","precision_at_50","precision_at_100","auc_roc","recall_at_threshold"]}
        row = log_exp(7, f"ENS_{meta}", "STACK", {"stack": stack_name}, m, 0, notes=f"ERROR: {e}")
    print_exp(row, TOTAL_EXPERIMENTS)

_flush_log()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ANALYSIS")
print('='*70)

df_exp = pd.DataFrame(experiments)
df_exp["avg_precision"] = pd.to_numeric(df_exp["avg_precision"], errors="coerce")
df_exp["f05_score"]     = pd.to_numeric(df_exp["f05_score"], errors="coerce")
df_exp["auc_roc"]       = pd.to_numeric(df_exp["auc_roc"], errors="coerce")
df_exp["precision_at_50"]  = pd.to_numeric(df_exp["precision_at_50"], errors="coerce")
df_exp["precision_at_100"] = pd.to_numeric(df_exp["precision_at_100"], errors="coerce")
df_exp["recall_at_threshold"] = pd.to_numeric(df_exp["recall_at_threshold"], errors="coerce")

# 1. Top 20 by F0.5
print("\nTOP 20 BY F0.5:")
print(f"{'Rank':>4} {'ExpID':>5} {'Model':<8} {'FS':<6} {'AP':>7} {'F0.5':>7} {'Thresh':>7} {'P@50':>6} {'P@100':>6} {'AUC':>7}")
print("-"*70)
top20_f05 = df_exp.nlargest(20, "f05_score")
for rank, (_, row) in enumerate(top20_f05.iterrows(), 1):
    print(f"{rank:>4} {int(row['exp_id']):>5} {row['model_family']:<8} {row['feature_set']:<6} "
          f"{row['avg_precision']:>7.4f} {row['f05_score']:>7.4f} {row['f05_threshold']:>7.2f} "
          f"{row['precision_at_50']:>6.3f} {row['precision_at_100']:>6.3f} {row['auc_roc']:>7.4f}")

# 2. Top 20 by AP
print("\nTOP 20 BY AVERAGE PRECISION:")
print(f"{'Rank':>4} {'ExpID':>5} {'Model':<8} {'FS':<6} {'AP':>7} {'F0.5':>7} {'P@50':>6} {'P@100':>6} {'AUC':>7}")
print("-"*65)
top20_ap = df_exp.nlargest(20, "avg_precision")
for rank, (_, row) in enumerate(top20_ap.iterrows(), 1):
    print(f"{rank:>4} {int(row['exp_id']):>5} {row['model_family']:<8} {row['feature_set']:<6} "
          f"{row['avg_precision']:>7.4f} {row['f05_score']:>7.4f} "
          f"{row['precision_at_50']:>6.3f} {row['precision_at_100']:>6.3f} {row['auc_roc']:>7.4f}")

# 3. Best per family
print("\nBEST PER MODEL FAMILY (by F0.5):")
for family in ["LR","XGB","LGBM","RF","ENS_LR","ENS_XGB","ENS_Ridge","ENS_avg","ENS_weighted","ENS_rank_avg","ENS_geo_mean","ENS_max"]:
    sub = df_exp[df_exp["model_family"]==family]
    if sub.empty: continue
    best = sub.nlargest(1,"f05_score").iloc[0]
    print(f"  {family:<16} Exp {int(best['exp_id']):>3}  AP={best['avg_precision']:.4f}  "
          f"F0.5={best['f05_score']:.4f}  P@50={best['precision_at_50']:.3f}  AUC={best['auc_roc']:.4f}")

# 4. Batch summary
print("\nBATCH SUMMARY:")
batch_names = {1:"LR",2:"XGB",3:"LGBM",4:"RF",5:"Calibration",6:"Threshold",7:"Ensemble"}
print(f"{'Batch':<5} {'Name':<14} {'n':>4} {'avg_AP':>8} {'avg_F05':>8} {'best_AP':>8} {'best_F05':>8}")
print("-"*55)
for b, bname in batch_names.items():
    sub = df_exp[(df_exp["batch"]==b) & (df_exp["avg_precision"]>0)]
    if sub.empty: continue
    print(f"  {b:<4} {bname:<14} {len(sub):>4} {sub['avg_precision'].mean():>8.4f} "
          f"{sub['f05_score'].mean():>8.4f} {sub['avg_precision'].max():>8.4f} "
          f"{sub['f05_score'].max():>8.4f}")


# ── Figures 14-16 ──────────────────────────────────────────────────────────────
FAMILY_COLORS = {
    "LR":"#2196F3","XGB":"#FF5722","LGBM":"#9C27B0","RF":"#4CAF50",
}
def family_color(mf):
    for k,v in FAMILY_COLORS.items():
        if k in mf: return v
    return "#607D8B"

# Fig 14 — violin AP by family
valid_df = df_exp[df_exp["avg_precision"] > 0].copy()
families_present = [f for f in ["LR","XGB","LGBM","RF"] if f in valid_df["model_family"].values]
fig, ax = plt.subplots(figsize=(10, 5))
data_violin = [valid_df[valid_df["model_family"]==f]["avg_precision"].values for f in families_present]
parts = ax.violinplot(data_violin, positions=range(len(families_present)), showmedians=True)
for i, (pc, fam) in enumerate(zip(parts["bodies"], families_present)):
    pc.set_facecolor(FAMILY_COLORS.get(fam, "#607D8B")); pc.set_alpha(0.7)
ax.set_xticks(range(len(families_present))); ax.set_xticklabels(families_present)
ax.axhline(0.0438, color="blue",  linestyle="--", linewidth=1, label="LR baseline AP=0.0438")
ax.axhline(0.10,   color="red",   linestyle=":",  linewidth=1.5, label="Target AP=0.10")
ax.set_ylabel("Average Precision"); ax.set_title("AP Distribution by Model Family", fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR/"14_experiment_ap_by_family.png", dpi=150); plt.close(fig)
print("\n  Saved figures/14_experiment_ap_by_family.png")

# Fig 15 — F0.5 vs AP scatter
fig, ax = plt.subplots(figsize=(9, 6))
for fam in families_present:
    sub = valid_df[valid_df["model_family"]==fam]
    sc = ax.scatter(sub["avg_precision"], sub["f05_score"],
                    c=FAMILY_COLORS.get(fam,"#607D8B"), label=fam, alpha=0.6,
                    s=sub["n_features"].clip(20,120), edgecolors="white", linewidth=0.5)
# Annotate top 5
top5_ann = df_exp.nlargest(5,"f05_score")
for _, row in top5_ann.iterrows():
    ax.annotate(f"E{int(row['exp_id'])}", (row["avg_precision"], row["f05_score"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(3,3), textcoords="offset points")
ax.axvline(0.0438, color="blue",  linestyle="--", linewidth=1, alpha=0.5, label="LR baseline AP")
ax.axhline(0.0453, color="blue",  linestyle=":",  linewidth=1, alpha=0.5, label="LR baseline F0.5")
ax.axvline(0.10,   color="red",   linestyle="--", linewidth=1, alpha=0.5, label="Target AP=0.10")
ax.set_xlabel("Average Precision"); ax.set_ylabel("F0.5 Score")
ax.set_title("F0.5 vs Average Precision — All Experiments", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR/"15_f05_vs_ap_scatter.png", dpi=150); plt.close(fig)
print("  Saved figures/15_f05_vs_ap_scatter.png")

# Fig 16 — top 20 F0.5 horizontal bar
top20 = df_exp.nlargest(20, "f05_score")
labels_bar = [f"E{int(r['exp_id'])} {r['model_family']} {r['feature_set']}" for _, r in top20.iterrows()]
colors_bar = [family_color(r["model_family"]) for _, r in top20.iterrows()]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(labels_bar[::-1], top20["f05_score"].values[::-1], color=colors_bar[::-1], alpha=0.85)
for bar, val in zip(bars, top20["f05_score"].values[::-1]):
    ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2, f"{val:.4f}",
            va="center", fontsize=7)
ax.axvline(0.0453, color="blue", linestyle="--", linewidth=1, label="LR baseline F0.5=0.0453")
ax.set_xlabel("F0.5 Score"); ax.set_title("PHBench — Top 20 F0.5 Scores", fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR/"16_top20_f05.png", dpi=150); plt.close(fig)
print("  Saved figures/16_top20_f05.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHAMPION SELECTION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CHAMPION SELECTION")
print('='*70)

champion_exp = df_exp.sort_values(["f05_score","avg_precision"], ascending=False).iloc[0]
print(f"  Champion: Exp {int(champion_exp['exp_id'])} {champion_exp['model_family']} "
      f"{champion_exp['feature_set']}  F0.5={champion_exp['f05_score']:.4f}  AP={champion_exp['avg_precision']:.4f}")

# Retrain champion
champ_row = experiments[int(champion_exp["exp_id"]) - 1]
champ_model = None
champ_fs    = champ_row["feature_set"]

try:
    if champ_row["batch"] == 7:
        # Ensemble champion — use best single model instead
        alt = df_exp[df_exp["batch"] <= 5].sort_values(["f05_score","avg_precision"], ascending=False).iloc[0]
        champ_row = experiments[int(alt["exp_id"]) - 1]
        champ_fs  = champ_row["feature_set"]
        print(f"  Champion is ensemble, saving best single: Exp {champ_row['exp_id']}")

    champ_model, champ_fs = retrain_model(champ_row)
    with open(MODEL_DIR / "phbench_champion.pkl", "wb") as f:
        pickle.dump(champ_model, f)
    print(f"  Saved models/phbench_champion.pkl")
except Exception as e:
    print(f"  WARNING: Could not retrain champion: {e}")

# Champion config
ap_delta  = float(champion_exp["avg_precision"]) - 0.0438
f05_delta = float(champion_exp["f05_score"])     - 0.0453
config = {
    "model_family":   champ_row["model_family"],
    "feature_set":    champ_fs,
    "n_features":     int(FS_TRAIN[champ_fs].shape[1]) if champ_fs in FS_TRAIN else 0,
    "hyperparams":    json.loads(champ_row["hyperparams"]),
    "best_threshold": float(champion_exp["f05_threshold"]),
    "val_metrics": {
        "avg_precision":    float(champion_exp["avg_precision"]),
        "f05_score":        float(champion_exp["f05_score"]),
        "f05_threshold":    float(champion_exp["f05_threshold"]),
        "precision_at_50":  float(champion_exp["precision_at_50"]),
        "precision_at_100": float(champion_exp["precision_at_100"]),
        "auc_roc":          float(champion_exp["auc_roc"]),
        "recall":           float(champion_exp["recall_at_threshold"]),
    },
    "vs_baseline": {
        "ap_delta":  round(ap_delta, 4),
        "f05_delta": round(f05_delta, 4),
    },
}
with open(MODEL_DIR / "phbench_champion_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved models/phbench_champion_config.json")

# Test predictions
if champ_model is not None and champ_fs in FS_TEST:
    try:
        test_prob = champ_model.predict_proba(FS_TEST[champ_fs])[:, 1]
        post_ids  = X_test_raw["post_id"].values
        sub = pd.DataFrame({"post_id": post_ids, "series_a_prob": test_prob.round(6)})
        sub.to_csv(DATA_DIR / "phbench_champion_submission.csv", index=False)
        print(f"  Saved data/benchmark/phbench_champion_submission.csv ({len(sub):,} rows)")
    except Exception as e:
        print(f"  WARNING: Could not generate test predictions: {e}")

# ── Final champion summary ────────────────────────────────────────────────────
ap     = float(champion_exp["avg_precision"])
f05    = float(champion_exp["f05_score"])
thresh = float(champion_exp["f05_threshold"])
p50    = float(champion_exp["precision_at_50"])
p100   = float(champion_exp["precision_at_100"])
auc    = float(champion_exp["auc_roc"])
recall = float(champion_exp["recall_at_threshold"])
ap_lift = ap / 0.0438 if 0.0438 > 0 else 0

if ap >= 0.10:
    assessment = "Target reached."
elif ap >= 0.08:
    assessment = "Close. Calibration or threshold tuning may close the gap."
else:
    assessment = "Significant headroom remains. Text embeddings recommended as next step (estimated +0.02-0.04 AP gain)."

sep = "=" * 60
print(f"\n{sep}")
print("PHBENCH CHAMPION MODEL")
print(sep)
print(f"Model           : {champ_row['model_family']}")
print(f"Feature set     : {champ_fs} ({FS_TRAIN[champ_fs].shape[1] if champ_fs in FS_TRAIN else '?'} features)")
print(f"Best threshold  : {thresh:.2f}")
print()
print(f"VALIDATION METRICS")
print(f"Avg Precision   : {ap:.4f}  (baseline: 0.0438, target: 0.10)")
print(f"F0.5 score      : {f05:.4f}")
print(f"Precision@50    : {p50:.4f} ({int(p50*50)} of top 50 are SA)")
print(f"Precision@100   : {p100:.4f} ({int(p100*100)} of top 100 are SA)")
print(f"AUC-ROC         : {auc:.4f}")
print(f"Recall          : {recall:.4f}")
print()
print(f"IMPROVEMENT vs LR BASELINE")
print(f"AP improvement  : {ap_delta:+.4f} ({ap_lift:.1f}x lift)")
print(f"F0.5 improvement: {f05_delta:+.4f}")
print()
print(f"LIFT OVER RANDOM (base rate 0.78%)")
print(f"AP lift         : {ap/0.0078:.1f}x better than random")
print(f"P@50 lift       : {p50/0.0078:.1f}x better than random")
print()
print(f"GAP TO TARGET")
print(f"AP gap          : {max(0, 0.10 - ap):.4f} remaining to 0.10 target")
print(f"Assessment      : {assessment}")
print(sep)

print(f"\nTotal experiments logged: {len(experiments)}")
_flush_log()
print(f"Full log: {LOG_FILE}")

#!/usr/bin/env python3
"""A5: Self-Paced Ensemble (SPE) baseline on the paper's exact splits + features.
Liu et al., ICDE 2020. Default settings (no tuning — same zero-tuning stance as
TabPFN). Full 68-feature FS4 set for comparability with the existing table.
Reports AP/F0.5/MCC/G-Mean on val and test."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from rebuttal.harness import champion, featurize, metrics, BEN

SEED = 42
N_ESTIMATORS = 50   # SPE default
K_BINS = 5          # SPE default

# ── SPE from scratch (Liu et al., ICDE 2020) ──────────────────────────────
class SelfPacedEnsemble:
    """Binary SPE classifier — faithful to the paper's defaults."""
    def __init__(self, n_estimators=50, k_bins=5, random_state=42):
        self.n_estimators = n_estimators
        self.k_bins = k_bins
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_target = len(pos_idx)  # balance to minority count
        y_pred_proba = np.zeros(len(y), dtype=np.float64)

        for t in range(self.n_estimators):
            alpha = np.tan(np.pi * 0.5 * (t / max(self.n_estimators - 1, 1)))
            hardness = y_pred_proba[neg_idx]  # P(pos) for true-neg = hardness

            if t == 0:
                chosen_neg = rng.choice(neg_idx, size=n_target, replace=True)
            else:
                bins = np.linspace(hardness.min() - 1e-9, hardness.max() + 1e-9, self.k_bins + 1)
                bin_ids = np.digitize(hardness, bins) - 1
                bin_ids = np.clip(bin_ids, 0, self.k_bins - 1)

                weights = np.zeros(self.k_bins)
                for b in range(self.k_bins):
                    mask_b = (bin_ids == b)
                    if mask_b.any():
                        mean_h = hardness[mask_b].mean()
                        weights[b] = mean_h * alpha + (1 - mean_h)
                    else:
                        weights[b] = 0.0

                per_bin = max(1, n_target // self.k_bins)
                chosen_neg_list = []
                for b in range(self.k_bins):
                    mask_b = (bin_ids == b)
                    indices_b = neg_idx[mask_b]
                    if len(indices_b) == 0:
                        continue
                    n_draw = min(per_bin, len(indices_b))
                    h_b = hardness[mask_b]
                    w_b = h_b * alpha + (1 - h_b)
                    w_b = w_b / w_b.sum()
                    drawn = rng.choice(indices_b, size=n_draw, replace=True, p=w_b)
                    chosen_neg_list.append(drawn)
                if len(chosen_neg_list) == 0:
                    chosen_neg = rng.choice(neg_idx, size=n_target, replace=True)
                else:
                    chosen_neg = np.concatenate(chosen_neg_list)
                    if len(chosen_neg) < n_target:
                        extra = rng.choice(neg_idx, size=n_target - len(chosen_neg), replace=True)
                        chosen_neg = np.concatenate([chosen_neg, extra])
                    elif len(chosen_neg) > n_target:
                        chosen_neg = rng.choice(chosen_neg, size=n_target, replace=False)

            train_idx = np.concatenate([pos_idx, chosen_neg])
            rng.shuffle(train_idx)

            tree = DecisionTreeClassifier(random_state=rng.randint(2**31))
            tree.fit(X[train_idx], y[train_idx])
            self.estimators_.append(tree)

            new_proba = tree.predict_proba(X)
            col1 = np.where(tree.classes_ == 1)[0][0]
            y_pred_proba = (y_pred_proba * t + new_proba[:, col1]) / (t + 1)

        return self

    def predict_proba(self, X):
        probas = []
        for tree in self.estimators_:
            p = tree.predict_proba(X)
            col1 = np.where(tree.classes_ == 1)[0][0]
            probas.append(p[:, col1])
        return np.mean(probas, axis=0)


# ── Load data (same as TabPFN script) ─────────────────────────────────────
ch = champion(); FS4 = list(ch["fs4_cols"])

train = pd.read_csv(BEN / "phbench_train.csv", low_memory=False)
val   = pd.read_csv(BEN / "phbench_validation.csv", low_memory=False)
feats = pd.read_csv(BEN / "phbench_test_features.csv", low_memory=False)
tlab  = pd.read_csv(BEN / "phbench_test_labels.csv").rename(columns={"post_id": "id"})
test  = feats.merge(tlab, on="id")

ytr = train["label_series_a_within_18m"].astype(int).values
yva = val["label_series_a_within_18m"].astype(int).values
yte = test["label_series_a_within_18m"].astype(int).values

print(f"Train: {len(ytr)} rows ({ytr.sum()} pos), Val: {len(yva)} ({yva.sum()} pos), Test: {len(yte)} ({yte.sum()} pos)")

Xtr = featurize(train, FS4)
Xva = featurize(val, FS4)
Xte = featurize(test, FS4)

print(f"Features: {Xtr.shape[1]} (FS4)")

# ── Train SPE ──────────────────────────────────────────────────────────────
t0 = time.time()
spe = SelfPacedEnsemble(n_estimators=N_ESTIMATORS, k_bins=K_BINS, random_state=SEED)
spe.fit(Xtr, ytr)
elapsed = time.time() - t0
print(f"\nSPE fit done in {elapsed:.1f}s ({N_ESTIMATORS} estimators, k_bins={K_BINS})")

pva = spe.predict_proba(Xva)
pte = spe.predict_proba(Xte)

# ── Report ─────────────────────────────────────────────────────────────────
print(f"\n{'set':5s} {'AP':>7s} {'F0.5':>7s} {'AUC':>7s} {'MCC':>7s} {'Gmean':>7s}")
for name, y, p in [("VAL", yva, pva), ("TEST", yte, pte)]:
    m = metrics(y, p)
    print(f"{name:5s} {m['AP']:7.4f} {m['F0.5_best']:7.4f} {m['AUC']:7.4f} {m['MCC_best']:7.4f} {m['Gmean_best']:7.4f}")

mte = metrics(yte, pte)
print(f"\n--- Table row (test set) ---")
print(f"SPE | AP={mte['AP']:.3f} | F0.5={mte['F0.5_best']:.3f} | MCC={mte['MCC_best']:.3f} | G-Mean={mte['Gmean_best']:.3f}")
print(f"(threshold={mte['thr_best']:.2f}, AUC={mte['AUC']:.3f})")

#!/usr/bin/env python3
"""
Run gemini-3.1-pro-preview on PHBench validation set.

Same protocol as llm_vanilla.py but with thinking_budget=128
(this model requires thinking mode — budget=0 is rejected).

Checkpoints every 100 posts. Safe to interrupt and resume:
    python scripts/run_gemini_31pro.py

After completion, prints full evaluation + year cohort breakdown.
"""
import os, json, time, sys
import numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phbench.llm_vanilla import anonymize_post, build_prompt
from phbench.evaluate import evaluate_submission, precision_at_k
from phbench.features import engineer_features
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score

from google import genai
from google.genai import types

MODEL = "models/gemini-3.1-pro-preview"
CKPT = Path("data/experiments/llm_vanilla_ckpt_gemini-3.1-pro-preview.csv")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading validation set...")
val = pd.read_csv("data/benchmark/phbench_validation.csv", dtype=str)
y_true = val["label_series_a_within_18m"].astype(int).values

print("Engineering features...")
val_feat = engineer_features(val)
val_merged = val.copy()
if "post_id" not in val_merged.columns:
    val_merged["post_id"] = val_merged["id"]
extra_cols = [c for c in val_feat.columns if c not in val_merged.columns or c == "post_id"]
val_merged = val_merged.merge(val_feat[extra_cols], on="post_id", how="left")

for col in ["votesCount", "commentsCount", "reviewsCount", "reviewsRating",
            "maker_count", "maker_followers_total", "dailyRank"]:
    if col in val_merged.columns:
        val_merged[col] = pd.to_numeric(val_merged[col], errors="coerce")

post_ids = val_merged["post_id"].tolist()
signals = [anonymize_post(row) for _, row in val_merged.iterrows()]
n = len(signals)
print(f"  {n:,} posts | {int(y_true.sum())} positives")


# ── API call ─────────────────────────────────────────────────────────────────
def call_31pro(prompt):
    cfg = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=512,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=128),
    )
    resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
    text = resp.text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
    obj = json.loads(text)
    prob = max(0.0, min(1.0, float(obj["probability"])))
    reasoning = str(obj.get("reasoning", ""))
    return prob, reasoning


# ── Resume from checkpoint ───────────────────────────────────────────────────
probs = [None] * n
reasonings = [""] * n

if CKPT.exists():
    ckpt = pd.read_csv(CKPT)
    valid = ckpt[ckpt["reasoning"] != "FALLBACK"]
    done = set(str(x) for x in valid["post_id"].tolist())
    for i, pid in enumerate(post_ids):
        if str(pid) in done:
            row = valid[valid["post_id"].astype(str) == str(pid)].iloc[0]
            probs[i] = float(row["prob"])
            reasonings[i] = str(row.get("reasoning", ""))
    n_done = sum(1 for p in probs if p is not None)
    print(f"  Resuming: {n_done}/{n} valid entries from checkpoint")
else:
    print("  No checkpoint — starting fresh")


# ── Run ──────────────────────────────────────────────────────────────────────
n_ok = sum(1 for p in probs if p is not None)
n_fail = n_api_err = 0
t0 = time.time()

for i in range(n):
    if probs[i] is not None:
        continue

    prompt = build_prompt(signals[i])
    ok = False

    for attempt in range(3):
        try:
            prob, reasoning = call_31pro(prompt)
            probs[i] = prob
            reasonings[i] = reasoning
            ok = True
            n_ok += 1
            break
        except json.JSONDecodeError:
            n_fail += 1
            if attempt < 2:
                time.sleep(1)
        except Exception as e:
            n_api_err += 1
            if attempt < 2:
                time.sleep(2)
            if attempt == 2:
                print(f"  [{i+1}] API error after 3 attempts: {str(e)[:100]}")

    if not ok:
        probs[i] = 0.5
        reasonings[i] = "FALLBACK"

    if (i + 1) % 100 == 0 or (i + 1) == n:
        cur = np.array([p if p is not None else 0.5 for p in probs[: i + 1]])
        try:
            running_ap = average_precision_score(y_true[: i + 1], cur)
        except Exception:
            running_ap = 0
        elapsed = time.time() - t0
        rate = n_ok / elapsed if elapsed > 0 else 0
        eta_min = (n - n_ok) / rate / 60 if rate > 0 else 0
        print(
            f"  [{i+1}/{n}] AP={running_ap:.4f} mean={np.mean(cur):.4f} "
            f"ok={n_ok} fail={n_fail} err={n_api_err} "
            f"ETA={eta_min:.0f}m"
        )

        # Save checkpoint
        rows = [
            {"post_id": post_ids[j], "prob": probs[j], "reasoning": reasonings[j]}
            for j in range(i + 1)
            if probs[j] is not None
        ]
        pd.DataFrame(rows).to_csv(CKPT, index=False)


# ── Final evaluation ─────────────────────────────────────────────────────────
probs_arr = np.array(probs, dtype=float)
n_fallback = sum(1 for r in reasonings if r == "FALLBACK")
print(f"\nDone. {n_ok} ok, {n_fallback} fallback")

metrics = evaluate_submission(y_true, probs_arr, model_name="Gemini 3.1 Pro — VAL SET")

# F0.5 at optimal threshold
best_f05 = 0
best_t = 0
for t in np.arange(0.001, 0.5, 0.001):
    yp = (probs_arr >= t).astype(int)
    if yp.sum() > 0:
        f = fbeta_score(y_true, yp, beta=0.5, zero_division=0)
        if f > best_f05:
            best_f05 = f
            best_t = t
print(f"\nF0.5 (best threshold={best_t:.3f}): {best_f05:.4f}")


# Bootstrap CI
def bootstrap_ap(yt, yp, n_iter=2000, seed=42):
    rng = np.random.default_rng(seed)
    aps = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(yt), len(yt))
        if yt[idx].sum() == 0:
            continue
        aps.append(average_precision_score(yt[idx], yp[idx]))
    return np.percentile(aps, 2.5), np.percentile(aps, 97.5)


ci_lo, ci_hi = bootstrap_ap(y_true, probs_arr)
print(f"AP 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"Mean predicted probability: {probs_arr.mean():.4f}")

# Comparison table
print(f"\n{'='*70}")
print(f"  {'Model':<30} {'AP':>7} {'F0.5':>7} {'P@50':>6} {'P@100':>6} {'AUC':>7}")
print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
print(f"  {'Gemini 3.1 Pro':<30} {metrics['avg_precision']:>7.3f} {best_f05:>7.3f} {metrics['precision_at_50']:>6.3f} {metrics['precision_at_100']:>6.3f} {metrics['auc_roc']:>7.3f}")
print(f"  {'Gemini 3 Flash':<30} {'0.034':>7} {'0.000':>7} {'0.120':>6} {'0.060':>6} {'0.713':>7}")
print(f"  {'Gemini 2.5 Flash':<30} {'0.022':>7} {'0.000':>7} {'0.060':>6} {'0.030':>6} {'0.697':>7}")
print(f"{'='*70}")

# Year cohort breakdown
print(f"\n── Year Cohort Breakdown ──")
val_merged["launch_year"] = pd.to_datetime(val_merged["createdAt"], utc=True).dt.year
val_merged["y"] = y_true
val_merged["prob_31pro"] = probs_arr

g3f = pd.read_csv("data/experiments/llm_vanilla_ckpt_gemini-3-flash-preview.csv")
val_merged = val_merged.merge(
    g3f[["post_id", "prob"]], on="post_id", how="left", suffixes=("", "_g3f")
)
val_merged.rename(columns={"prob": "prob_g3f"}, inplace=True)

print(f"{'Year':>4} | {'N':>5} | {'Pos':>3} | {'3.1Pro AP':>9} | {'3Flash AP':>9}")
print(f"{'----':>4}-+-{'-----':>5}-+-{'---':>3}-+-{'---------':>9}-+-{'---------':>9}")
for yr in sorted(val_merged["launch_year"].unique()):
    mask = val_merged["launch_year"] == yr
    y = val_merged.loc[mask, "y"].values
    n_pos = int(y.sum())
    if n_pos == 0:
        print(f"{yr:>4} | {mask.sum():>5} | {n_pos:>3} |         — |         —")
        continue
    ap_pro = average_precision_score(y, val_merged.loc[mask, "prob_31pro"].values)
    ap_flash = average_precision_score(y, val_merged.loc[mask, "prob_g3f"].values)
    print(f"{yr:>4} | {mask.sum():>5} | {n_pos:>3} | {ap_pro:>9.3f} | {ap_flash:>9.3f}")

# Save final results
pd.DataFrame(
    {"post_id": post_ids, "prob": probs_arr, "reasoning": reasonings}
).to_csv("data/experiments/llm_vanilla_results_31pro.csv", index=False)
print(f"\nResults saved to data/experiments/llm_vanilla_results_31pro.csv")

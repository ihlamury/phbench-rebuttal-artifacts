# PHBench — Evaluation & Orchestration Scripts

This repository publishes the evaluation and experiment-orchestration scripts for the
PHBench benchmark, provided verbatim so that every headline number in the paper can be
independently regenerated from the public data and public model artifacts.

These three scripts are the pipeline drivers referenced by the paper. They are published
here in addition to the main code repository (data collection, feature engineering, split
construction, and configs/outputs), which is available separately.

## Contents

| File | What it produces |
|---|---|
| `experiments.py` | The full ML configuration sweep with F0.5 threshold selection. Writes `experiment_log.csv` and the experiment IDs referenced by the champion configs (exp138, exp139, exp115, exp21, exp20). |
| `followup_experiments.py` | Isotonic calibration producing the `ENS_ISO` component and the Top-3 ensemble champion, plus threshold optimization. |
| `run_gemini_31pro.py` | The Gemini 3.1 Pro validation run (`thinking_budget=128`), checkpointed and resumable. |

All three are published **verbatim** — no edits, renames, or refactors. Integrity is
pinned by `SHA256SUMS`.

## Integrity

Verify the published files against the manifest:

```bash
shasum -a 256 -c SHA256SUMS
```

Expected:

```
5f8fb9a314285de7b04303693a282610a8c99cac68db18c5e342629ee6fa8b23  experiments.py
a67d37395453691d54d321c342734e7a42d639cfc483f2552e479de97f8c3e40  followup_experiments.py
f503a7bef8b77a09af3bd3fc9404673e7fe24f446d2dbe83a0ab1b6f79820498  run_gemini_31pro.py
```

## Regeneration

With the public dataset splits and the public model artifacts (saved models, splits, and
prediction arrays), these scripts regenerate the paper's headline numbers:

| Number | Paper | Regenerated |
|---|---|---|
| Champion test AP | 0.037 | 0.0373 (exact, from saved probability array) |
| Champion test F0.5 | 0.097 | 0.0965 (exact) |
| Champion test AP (from raw code) | 0.037 | 0.0368 (corr 0.998 with saved array) |
| Validation LLM (Gemini 3 Flash AP) | 0.034 | 0.0342 (exact) |

The champion reproduces exactly from its saved probability array and to correlation 0.998
when recomputed end-to-end from the feature code and saved models.

The exact-from-array numbers regenerate using only the public prediction arrays and labels.
The raw-code regeneration path (correlation 0.998) additionally requires the feature
pipeline, split construction, and model artifacts from the main public repository:
https://github.com/ihlamury/phbench

## Requirements

Python 3.11+, with `scikit-learn`, `xgboost`, `lightgbm`, `numpy`, `pandas`. The Gemini
runner additionally needs `google-genai` and a `GEMINI_API_KEY` environment variable. No
credentials are embedded in any script; all keys are read from the environment.

## License

Released for research reproducibility. See the main PHBench project for dataset terms.

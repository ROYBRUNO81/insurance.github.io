# Car Insurance Claim Prediction

This project explores two car-insurance datasets and builds models to predict whether a policyholder will file a claim. The work is organized in two Jupyter notebooks:

- `Portfolio_Project.ipynb` — Vehicle/specs dataset (highly imbalanced target)
- `Car_Claims_Analysis.ipynb` — Driver-profile dataset (balanced target)

## Datasets

1) Vehicle/specs dataset (imbalanced)
- Features: vehicle specs, equipment flags, regions, etc.
- Target: `claim_status` (≈6.4% positives)

2) Driver-profile dataset (balanced)
- Features: demographics, driving history, credit and mileage
- Target: `OUTCOME` (~31% positives)

## Approach

- Keep EDA visuals on raw data.
- Split train/test before any target-aware transforms.
- Encoding
  - High-cardinality location → target encoding with cross‑fitting
    - `region_code` in the first dataset
    - `POSTAL_CODE` in the second dataset
  - Remaining categoricals → One‑Hot Encoding.
- Missing values
  - In pipelines only (not pre-filled in data)
  - Numeric: median by default; for bell‑shaped variables like CREDIT_SCORE we use mean imputation
  - Categorical: most_frequent
- Threshold selection via the Precision–Recall curve 

## Models and key results

1) Vehicle/specs (Portfolio_Project.ipynb)
- Baselines
  - Logistic Regression (class_weight='balanced'): ROC‑AUC ≈ 0.62, PR‑AUC ≈ 0.09
  - Random Forest (tuned, class_weight='balanced'): ROC‑AUC ≈ 0.66, PR‑AUC ≈ 0.11
- Notes
  - Strong class imbalance makes precision low at useful recall.
  - After correlation analysis, near‑deterministic equipment flags were dropped; `region_code` was target‑encoded.

2) Driver profile (Car_Claims_Analysis.ipynb)
- Logistic Regression (balanced data): ROC‑AUC 0.906, PR‑AUC 0.812, Accuracy 0.842
  - Confusion (0.5): TN=1224, FP=149, FN=168, TP=459
- Random Forest: ROC‑AUC 0.897, PR‑AUC 0.787, Accuracy 0.829
  - Confusion (0.5): TN=1193, FP=180, FN=162, TP=465
- Takeaway
  - LR has better ranking (PR/ROC‑AUC), higher precision, fewer false positives.
  - RF finds slightly more true claims (465 vs 459) with more false alarms.

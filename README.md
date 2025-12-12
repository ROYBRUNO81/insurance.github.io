# 🚗 Car Insurance Claim Prediction: Dual Dataset Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **A comprehensive binary classification analysis predicting insurance claims across two distinct datasets: vehicle specifications (imbalanced) and driver profiles (balanced)**

[View Vehicle Analysis](https://github.com/ROYBRUNO81/insurance.github.io/blob/main/projects/insurance-claims/Portfolio_Project.ipynb) | [View Driver Analysis](https://github.com/ROYBRUNO81/insurance.github.io/blob/main/projects/insurance-claims/Car_Claims_Analysis.ipynb) | [Dataset Source1](https://www.kaggle.com/datasets/litvinenko630/insurance-claims/data) | [Dataset Source2](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data)

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Datasets](#datasets)
- [Technical Approach](#technical-approach)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)

---

## 🎯 Project Overview

This project tackles the car insurance claim prediction problem from two complementary angles: **vehicle/equipment characteristics** and **driver behavioral profiles**. By analyzing both datasets, we provide insurers with actionable insights into risk factors that span both the insured asset and the policyholder's driving history.

### **Business Problem**
Insurance companies need accurate claim prediction models to:
- Set appropriate premium rates
- Identify high-risk policyholders early
- Optimize loss ratios and profitability
- Improve underwriting decisions

However, claim datasets often suffer from severe class imbalance (only 6-7% of policies result in claims), making traditional accuracy metrics misleading and requiring specialized modeling techniques.

### **Objectives**
- Build robust binary classifiers for two distinct insurance contexts
- Handle severe class imbalance using sampling strategies and threshold optimization
- Engineer features that capture both vehicle risk and driver behavior
- Compare linear (Logistic Regression) vs. tree-based models (Random Forest, XGBoost)
- Identify the strongest predictors of claim likelihood across both datasets

---

## ✨ Key Features

### **Dual Dataset Analysis**
- ✅ **Portfolio_Project.ipynb**: Vehicle specifications dataset (58K+ records, 6.4% positive rate)
- ✅ **Car_Claims_Analysis.ipynb**: Driver profile dataset (10K records, 31% positive rate)
- ✅ Contrasting imbalanced vs. balanced classification scenarios

### **Advanced Preprocessing Pipeline**
- 🔧 **Leakage-aware encoding**:
  - Cross-fitted target encoding for high-cardinality location features
  - Train-test split **before** any target-dependent transforms
  - One-Hot Encoding for remaining categoricals
- 🔧 **Strategic missing value imputation**:
  - Mean imputation for normally-distributed features (e.g., CREDIT_SCORE)
  - Median imputation for skewed numerics
  - Most-frequent imputation for categoricals

### **Comprehensive EDA & Feature Engineering**
- 📊 Correlation analysis (Cramér's V for categoricals, Pearson for numerics)
- 📊 Outlier detection and treatment (Tukey fences for vehicle_age)
- 📊 Feature pruning based on near-deterministic relationships
- 📊 Risk visualization across categorical features

### **Model Development & Tuning**
- 🤖 Baseline: Logistic Regression with class weighting
- 🤖 Ensemble: Random Forest with GridSearchCV optimization
- 🤖 Gradient Boosting: XGBoost with scale_pos_weight tuning
- 🤖 Threshold selection via Precision-Recall curve analysis

---

## 📊 Datasets

### **Dataset 1: Vehicle/Equipment Specifications (Imbalanced)**
**Source:** [Kaggle Insurance Claims Dataset](https://www.kaggle.com/datasets/litvinenko630/insurance-claims/data)  
**Size:** 58,592 records → 58,323 after outlier filtering  
**Target:** `claim_status` (≈6.4% positive class)  
**Features:** 41 original columns including:

| Feature Category | Examples |
|------------------|----------|
| **Policy Info** | subscription_length, region_code, region_density |
| **Vehicle Specs** | vehicle_age, segment, fuel_type, displacement, cylinder |
| **Safety Features** | airbags, ncap_rating, is_esc, is_tpms, is_brake_assist |
| **Equipment Flags** | is_parking_camera, is_central_locking, is_power_steering |
| **Engine Details** | max_torque, max_power, engine_type, transmission_type |

*Note: Many equipment flags showed near-perfect correlation (Cramér's V > 0.95) and were pruned*

---

### **Dataset 2: Driver Profile & Behavior (Balanced)**
**Source:** [Kaggle Car Insurance Data](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data)  
**Size:** 10,000 records  
**Target:** `OUTCOME` (31.3% positive class)  
**Features:** 19 original columns including:

| Feature Category | Examples |
|------------------|----------|
| **Demographics** | AGE, GENDER, RACE, EDUCATION, INCOME, MARRIED, CHILDREN |
| **Financial** | CREDIT_SCORE (9.8% missing) |
| **Driving Profile** | DRIVING_EXPERIENCE, ANNUAL_MILEAGE (9.6% missing) |
| **Vehicle Info** | VEHICLE_OWNERSHIP, VEHICLE_YEAR, VEHICLE_TYPE |
| **Risk Indicators** | SPEEDING_VIOLATIONS, DUIS, PAST_ACCIDENTS |
| **Geographic** | POSTAL_CODE (high cardinality → target encoded) |
| **Purchase** | BOOKING_SOURCE |

*Note: GENDER and RACE excluded from training to avoid discriminatory modeling*

---

## 🔬 Technical Approach

### **1. Data Preprocessing Strategy**

#### **Common Pipeline (Both Datasets)**
```python
# Preprocessing order (critical for avoiding leakage):
1. Drop identifiers and redundant columns
2. Handle outliers (Tukey method for vehicle_age)
3. Train-test split (stratified by target)
4. Target encoding for high-cardinality categoricals (cross-fitted)
5. One-Hot Encoding for remaining categoricals
6. Imputation and scaling in sklearn Pipeline
```

#### **Dataset-Specific Considerations**

**Vehicle Dataset:**
- Dropped near-deterministic equipment flags (Cramér's V > 0.8)
- Target-encoded `region_code` (high cardinality, geographic risk patterns)
- Filtered `vehicle_age` outliers beyond Q3 + 1.5×IQR (removed 269 records)
- Retained: fuel_type, transmission_type, selected safety features

**Driver Dataset:**
- Target-encoded `POSTAL_CODE` with smoothing=50
- Excluded sensitive features (GENDER, RACE) from training
- Mean imputation for CREDIT_SCORE (bell-shaped distribution)
- Median imputation for ANNUAL_MILEAGE (right-skewed)

---

### **2. Feature Engineering Highlights**

#### **Target Encoding (Cross-Fitted)**
```python
def target_encode_crossfit(X, y, col, n_splits=5, smoothing=50):
    # Stratified K-Fold to prevent leakage
    # Smoothed mean: (count × mean + smoothing × global) / (count + smoothing)
    # Produces numeric risk score per category
```

#### **Correlation-Driven Pruning**
- Computed **Cramér's V** for categorical pairs
- Computed **Pearson r** for numeric pairs
- Dropped features with r > 0.85 or Cramér's V > 0.90
- **Example**: `segment` perfectly predicted `engine_type`, `is_power_steering`, `rear_brakes_type` → kept `segment`, dropped others

---

### **3. Modeling Strategy**

We evaluated **multiple model architectures** with **class-imbalance handling** across both datasets:

#### **Baseline: Logistic Regression**
- **Driver Dataset**: `class_weight='balanced'`, `max_iter=2000`, `solver='lbfgs'`
- **Vehicle Dataset**: Same configuration
- **Purpose**: Establish linear baseline for comparison

#### **Ensemble: Random Forest**
- **Tuning**: GridSearchCV over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- **Scoring**: `average_precision` (PR-AUC) to optimize for imbalanced data
- **Config**: `class_weight='balanced'`, `n_jobs=-1`

#### **Gradient Boosting: XGBoost** (Vehicle Dataset Only)
- **Imbalance handling**: `scale_pos_weight = n_negative / n_positive`
- **Tuning**: GridSearchCV over `learning_rate`, `max_depth`, `n_estimators`, `subsample`
- **Regularization**: `min_child_weight`, `colsample_bytree`

---

### **4. Threshold Selection via Precision-Recall Curve**

For imbalanced problems, the default 0.5 threshold is often suboptimal. We analyzed:

1. **Default (0.5)**: Baseline confusion matrix
2. **Best F1**: Threshold maximizing F1-score on PR curve
3. **Precision-constrained**: Threshold achieving minimum precision (e.g., 0.25) with maximum recall
```python
prec, rec, thr = precision_recall_curve(y_test, probs)
f1_vals = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_threshold = thr[f1_vals.argmax()]
```

---

## 🏆 Results

### **Dataset 1: Vehicle/Equipment Specifications (Imbalanced)**

#### **Model Performance Comparison**

| Model | ROC-AUC | PR-AUC | Accuracy @ 0.5 | Log Loss | Notes |
|-------|---------|--------|----------------|----------|-------|
| **Logistic Regression** | 0.617 | ~0.09 | 0.559 | 0.675 | Baseline linear model |
| **Random Forest** | **0.660** | **0.109** | 0.732 | 0.517 | Best tuned ensemble ✅ |
| **XGBoost** | 0.669 | 0.115 | 0.582 @ 0.5 | 0.632 | Highest ROC-AUC but low precision |

#### **Best Model: Random Forest (Final)**
```
Configuration:
- n_estimators: 300
- max_depth: 12
- min_samples_leaf: 10
- max_features: 'sqrt'
- class_weight: 'balanced'

Performance @ threshold = 0.5:
✅ ROC-AUC: 0.660
✅ PR-AUC: 0.109
✅ Accuracy: 0.732
✅ Precision: 0.112 (112 claims per 1000 flagged)
✅ Recall: 0.461 (catches 46% of true claims)
✅ F1-Score: 0.181

Confusion Matrix (Test Set: 11,665 samples):
              Predicted
              No    Yes
Actual No   8194   2723
       Yes   403    345
```

#### **Key Findings (Vehicle Dataset)**

1. **Severe imbalance limits precision**  
   - Only 6.4% positive class makes high precision difficult
   - Trade-off: Catching 46% of claims requires flagging ~27% of policies

2. **Tree ensembles outperform linear models**  
   - RF achieved 7% higher ROC-AUC vs. Logistic Regression
   - Non-linear interactions (e.g., region × vehicle_age × segment) matter

3. **Equipment redundancy was massive**  
   - 20+ features dropped due to Cramér's V > 0.90
   - Example: `is_power_steering` perfectly predicted by `segment`

4. **XGBoost showed highest discrimination but lower precision**  
   - Best ROC-AUC (0.669) but struggled with precision at useful recall
   - RF better balanced precision-recall trade-off

---

### **Dataset 2: Driver Profile & Behavior (Balanced)**

#### **Model Performance Comparison**

| Model | ROC-AUC | PR-AUC | Accuracy @ 0.5 | Precision @ 0.5 | Recall @ 0.5 | F1-Score |
|-------|---------|--------|----------------|-----------------|--------------|----------|
| **Logistic Regression** | **0.906** | **0.812** | **0.842** | **0.755** | **0.732** | **0.743** ✅ |
| **Random Forest** | 0.897 | 0.787 | 0.829 | 0.721 | 0.742 | 0.731 |

#### **Best Model: Logistic Regression**
```
Configuration:
- class_weight: 'balanced'
- max_iter: 2000
- solver: 'lbfgs'

Performance @ threshold = 0.5:
✅ ROC-AUC: 0.906 (excellent discrimination)
✅ PR-AUC: 0.812 (strong ranking)
✅ Accuracy: 0.842
✅ Precision: 0.755 (75.5% of flagged cases are true claims)
✅ Recall: 0.732 (catches 73% of claims)
✅ F1-Score: 0.743

Confusion Matrix (Test Set: 2,000 samples):
              Predicted
              No    Yes
Actual No   1224    149
       Yes   168    459
```

#### **Key Findings (Driver Dataset)**

1. **Balanced data enables strong performance**  
   - 31% positive class allows much better precision (75.5% vs. 11.2%)
   - Both models achieve >82% accuracy

2. **Linear model wins on balanced data**  
   - Logistic Regression outperforms RF across all metrics
   - Driver risk appears approximately linear in feature space
   - Simpler model → better interpretability for stakeholders

3. **Precision-recall trade-off is favorable**  
   - LR: 149 false positives, 168 false negatives
   - RF: 180 false positives, 162 false negatives
   - LR finds 6 fewer true claims but creates 31 fewer false alarms

4. **Target-encoded POSTAL_CODE was crucial**  
   - Geographic risk patterns captured without sparse OHE
   - Cross-fitting prevented overfitting

---

### **Cross-Dataset Insights**

| Aspect | Vehicle Dataset (Imbalanced) | Driver Dataset (Balanced) |
|--------|------------------------------|---------------------------|
| **Best Model** | Random Forest | Logistic Regression |
| **Class Imbalance Impact** | Severe (6.4% positive) → low precision unavoidable | Moderate (31% positive) → high precision achievable |
| **Feature Space** | Non-linear (equipment × region interactions) | Approximately linear (additive risk factors) |
| **ROC-AUC** | 0.660 (moderate discrimination) | 0.906 (excellent discrimination) |
| **Precision @ 0.5** | 0.112 (poor) | 0.755 (strong) |
| **Business Action** | Need higher threshold or cost-benefit analysis | Can flag high-risk drivers confidently |

---

### **Model Comparison Visualization**
```
ROC-AUC Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vehicle Dataset (Imbalanced):
  Logistic Reg   ████████████        0.617
  Random Forest  █████████████       0.660 ⭐
  XGBoost        █████████████       0.669

Driver Dataset (Balanced):
  Logistic Reg   ██████████████████  0.906 ⭐
  Random Forest  █████████████████   0.897

Precision @ 0.5:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vehicle Dataset:
  Best (RF)      ██                  0.112

Driver Dataset:
  Best (LR)      ███████████████     0.755
```

---

## 🚀 Installation & Usage

### **Prerequisites**
```bash
Python 3.8+
pip install -r requirements.txt
```

### **Dependencies**
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
joblib>=1.0.0
```

### **Quick Start**
```bash
# Clone the repository
git clone git@github.com:ROYBRUNO81/insurance.github.io.git
cd car-insurance-claim-prediction

# Install dependencies
pip install -r requirements.txt

# Download datasets from Kaggle (see Dataset section)
# Place in project root:
#   - insurance_claims.csv
#   - Car_Insurance_Claim.csv

# Launch Jupyter Notebook
jupyter notebook
```

### **Running the Analyses**

#### **Vehicle Dataset Analysis**
```bash
# Open Portfolio_Project.ipynb
1. Load data (insurance_claims.csv)
2. Run EDA cells (correlation analysis, outlier detection)
3. Execute preprocessing pipeline
4. Train models (LR, RF, XGBoost)
5. Evaluate threshold strategies
6. Save final model artifacts
```

#### **Driver Dataset Analysis**
```bash
# Open Car_Claims_Analysis.ipynb
1. Load data (Car_Insurance_Claim.csv)
2. Run EDA cells (risk distributions, missingness analysis)
3. Execute preprocessing pipeline
4. Train models (LR, RF)
5. Compare confusion matrices
6. Review final recommendations
```

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn (LogisticRegression, RandomForestClassifier, Pipeline, ColumnTransformer) |
| **Gradient Boosting** | XGBoost |
| **Statistical Analysis** | SciPy (chi2_contingency, Cramér's V) |
| **Model Persistence** | joblib |
| **Development** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

---

## 👥 Contributors

**Bruno Ndiba Mbwaye Roy**

*AI/ML Portfolio Project*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

</div>

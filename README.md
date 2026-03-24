# Loan Default Risk Prediction Model

This is a machine learning model that predicts whether a borrower will default on a loan, using real credit data from Kaggle's "Give Me Some Credit" competition, which contains anonymized borrower records . I built it to demonstrate classification modeling and feature importance analysis.

---

## Overview

In this project, I used  a **Random Forest Classifier** to predict loan default risks. It analyzes borrower financial profiles like credit utilization, debt ratio, income, payment history to determine the likelihood of default.


---

##  What It Does

| Input | Output |
|-------|--------|
| Credit utilization, debt ratio, income, age, payment history | **Default Risk** (Yes/No) + Probability |

---

## Key Findings

The model identifies which factors most influence default risk:

| Feature | Importance |
|---------|------------|
| Credit Utilization | Highest |
| Debt Ratio | 2nd Highest |
| Monthly Income | 3rd Highest |
| Age | Important |
| Past Delinquencies (90+ days) | Significant |

The features align with real-world credit risk assessment where borrowers with high credit usage, high debt, or past payment issues are more likely to default.

---

## Dataset

**Source:** Kaggle – [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

**Size:** ~250,000 borrower records

**Target:** `SeriousDlqin2yrs` (1 = defaulted, 0 = paid back)

**Features:**
- Credit utilization (balance / limit)
- Debt ratio (debt / income)
- Monthly income
- Age
- Past payment history (30-59 days, 60-89 days, 90+ days late)
- Number of open credit lines
- Number of real estate loans
- Number of dependents

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Python | Core programming |
| pandas | Data manipulation |
| scikit-learn | Random Forest Classifier |
| matplotlib | Visualization |
| seaborn | Confusion matrix heatmap |

---

## How It Works

1. **Load Data** – Real Kaggle dataset
2. **Clean Data** – Handle missing values, replace anomalies (98 values)
3. **Train Model** – Random Forest Classifier (100 trees)
4. **Evaluate** – Accuracy score, confusion matrix
5. **Interpret** – Feature importance chart shows risk drivers
6. **Visualize** – Default rate by age group

---

## Visual Outputs

The model generates three images:

| Image | What It Shows |
|-------|---------------|
| `feature_importance.png` | Which factors matter most |
| `confusion_matrix.png` | How many predictions were correct |
| `default_by_age.png` | Default rates across age groups |

---

## Model Performance

- **Accuracy:** ~93-94% on test data
- **Key Insight:** Credit utilization is the strongest predictor where a borrower using more than 70% of available credit is significantly higher risk.

---

## Files Included

| File | Description |
|------|-------------|
| `loan_default_model.py` | Main script |
| `cs-training.csv` | Dataset (real Kaggle data) |
| `feature_importance.png` | Feature importance chart |
| `confusion_matrix.png` | Confusion matrix |
| `default_by_age.png` | Age group risk chart |
| `.gitignore` | Ignores unnecessary files |
| `README.md` | This file |

---

## Real-World Application

Banks and lenders use models like this to:

- Assess loan application risk
- Set interest rates
- Determine credit limits
- Comply with regulatory requirements

The feature importance analysis helps explain why a decision was made — which is critical for transparency and compliance.

---

## References

- Dataset: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle)


---

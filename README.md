# Car Insurance Claim Prediction

A machine learning classification project that predicts whether a customer will file a car insurance claim, based on demographic, financial, vehicle and driving history attributes.

## Project Overview

In motor insurance, predicting claim risk helps with underwriting and premium pricing. The target is `CLAIM_FLAG`:

* `0`: Customer did not file a claim
* `1`: Customer filed a claim

The notebook covers data auditing, cleaning, exploratory data analysis, preprocessing, baseline models, feature selection, hyperparameter tuning, final evaluation on a held-out test set, learning curves and a bias-variance analysis. Every step is tracked with MLflow.

## Models

Dummy baseline, Logistic Regression, Decision Tree, KNN, SVM and Gradient Boosting (scikit-learn).

## Key Points

* **Class imbalance:** about 73% of customers did not claim and 27% did. Models are compared with ROC-AUC and F1, not accuracy.
* **Data leakage:** `CLM_AMT` (claim payout) was dropped because it is only known after a claim happens.
* **Messy data:** currency stored as text, `z_` label prefixes, impossible values and missing values were fixed. Missing values are imputed inside the pipeline.
* **Feature selection:** variance threshold, mutual information, SFS and SBS (mlxtend) and L1 regularisation.
* **Result:** the tuned Gradient Boosting model reaches a test ROC-AUC of 0.832 and catches about 78% of the claimants.

## Project Structure

```text
ML_Classification_Project/
├── data/
│   └── car_insurance_claim.csv               # Dataset
├── mlruns/                                   # Local MLflow tracking logs (created when the notebook runs)
├── Car_Insurance_Claim_Classification.ipynb  # Main notebook
├── .gitignore
└── README.md
```

## Dataset Summary

* **Records:** 10,302 rows (10,301 after removing one duplicate)
* **Features:** 23 after cleaning (numeric, binary, ordinal and nominal)
* **Target:** `CLAIM_FLAG` (binary: 0 or 1)
* **Source:** https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data

## Getting Started

### 1. Installation

```bash
pip install -U scikit-learn mlflow mlxtend pandas numpy matplotlib seaborn
```

### 2. Run the Notebook

Open `Car_Insurance_Claim_Classification.ipynb` and run all cells in order. A full run takes about 10 minutes.

### 3. MLflow Experiment Tracking

From the project folder:

```bash
# Windows (PowerShell)
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --backend-store-uri file:./mlruns
```

Then open `http://127.0.0.1:5000`.

## Author

* **Moheeb Abusamra**

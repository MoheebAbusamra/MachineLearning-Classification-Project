# Car Insurance Claim Prediction

A machine learning classification project that predicts whether a customer will submit an auto insurance claim based on demographic, financial, vehicle, and driving history attributes.

## Project Overview

In motor insurance, assessing claim risk accurately is essential for underwriting and premium pricing. This project builds and evaluates classification models to predict the binary target `CLAIM_FLAG`:

* `0`: Customer did not file a claim
* `1`: Customer filed a claim

The workflow covers data auditing, exploratory data analysis, cleaning and preprocessing, feature engineering, and experiment tracking using MLflow.

## Key Challenges and Solutions

* **Target Imbalance:** About 73% of customers did not file a claim while 27% filed a claim. Evaluation focuses on metrics like ROC-AUC and F1-score rather than accuracy alone.
* **Data Leakage Prevention:** The column `CLM_AMT` (claim payout amount) was dropped because it is only known after a claim occurs.
* **Messy Real-World Data:** 
  * Currency fields formatted as text strings were cleaned into numeric values.
  * Inconsistent and impossible values were audited and corrected.
  * Missing values were evaluated and handled using imputation rather than row deletion to preserve data volume.
* **Feature Categorization:** Attributes are organized into numeric, binary, ordinal (Education), and nominal (Occupation, Car Type) groups with tailored transformations for each group.

## Project Structure

```text
ML_Classification_Project/
│
├── data/
│   └── car_insurance_claim.csv               # Dataset
├── mlruns/                                   # Local MLflow tracking logs
├── Car_Insurance_Claim_Classification.ipynb  # Main analysis and modeling notebook
├── .gitignore                                # Git ignore rules
└── README.md                                 # Project documentation
```

## Dataset Summary

* **Records:** 10,302 rows
* **Features:** Demographic, driving record, financial, and vehicle attributes
* **Target:** `CLAIM_FLAG` (binary: 0 or 1)

## Getting Started

### 1. Prerequisites

Make sure you have Python 3.9 or newer installed on your machine.

### 2. Installation

Clone or download this repository, then install the required dependencies:

```bash
pip install -U scikit-learn mlflow mlxtend pandas numpy matplotlib seaborn
```

### 3. Run the Notebook

Launch JupyterLab or Jupyter Notebook:

```bash
jupyter notebook Car_Insurance_Claim_Classification.ipynb
```

Run the cells sequentially to reproduce the exploratory analysis, data cleaning, and experiments.

### 4. MLflow Experiment Tracking

To view and compare experiment runs, metrics, and parameters in the MLflow dashboard, run the following command in the project root:

```bash
mlflow ui
```

Then open your browser and navigate to `http://localhost:5000`.

## Author

* **Moheeb Abusamra** 


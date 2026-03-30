# 🛡️ Modular Fraud Detection Pipeline with XAI

A Machine Learning pipeline designed to identify fraudulent credit card transactions. This project focuses on **feature engineering**, **modular software design**, and **Explainable AI (XAI)** to bridge the gap between "black-box" models and actionable business insights.

## 🚀 Project Highlights
* **Modular Architecture:** Clean separation of concerns (Features, Preprocessing, Training, Evaluation) following production-ready standards.
* **Feature Engineering:** Goes beyond raw PCA components by implementing behavioral features:
    * `amount_ratio` & `amount_zscore`: Captures spending "shocks" by comparing current transactions to 30-day rolling windows.
    * `time_delta`: Detects high-frequency "brute-force" attack patterns.
    * `feat_dist`: Measures abrupt shifts in a user’s behavioral profile.
* **Explainable AI (SHAP):** Uses Shapley Additive Explanations to visualize exactly *why* the model flags a transaction as fraudulent.
* **Imbalanced Data Handling:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to train effectively on datasets where fraud represents <0.2% of cases.

## 🏗️ Directory Structure
```text
FRAUD_DETECTION/
├── Features/               # Feature Factory - Custom behavioral logic
├── src/                    # Core Pipeline (Loader, Preprocessor, Trainer, Evaluator)
├── models/                 # Serialized .pkl models and feature metadata
├── data/                   # Dataset storage (creditcard.csv)
└── __main__.py             # Main entry point to run the full pipeline
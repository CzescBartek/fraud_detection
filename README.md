# 🛡️ Real-Time Credit Card Fraud Detection System

A Machine Learning pipeline and REST API designed to detect fraudulent transactions using behavioral analysis and Explainable AI (XAI).

## 🌟 Key Features
* **End-to-End Pipeline:** From raw data to a live production API.
* **Feature Engineering:**
    * `feat_dist`: High-dimensional Euclidean distance measuring behavioral shifts (Top 5 Feature).
    * `amount_ratio` & `amount_zscore`: Real-time anomaly detection in transaction values.
    * `time_delta`: Captures high-frequency attack patterns.
* **Explainable AI (XAI):** Integrated **SHAP** (Global & Local) to provide full transparency for every "BLOCK" decision.
* **Production-Ready API:** Built with **FastAPI**, featuring automated Swagger documentation and input validation.

## 🏗️ System Architecture
The project follows a modular structure for scalability:
* `Features/`: Custom logic for behavioral feature extraction.
* `src/`: Core engine (Preprocessing, Training, Evaluation, XAI).
* `api/`: REST API implementation for real-time inference.
* `models/`: Persistent storage for trained models, scalers, and metadata.

## 📊 Model Performance
* **Algorithm:** Random Forest Classifier (optimized with SMOTE).
* **Metric:** **Average Precision (AP) = 0.77**.
* **Interpretability:** The model prioritizes behavioral consistency (`feat_dist`) and transaction context over raw values, significantly reducing False Positives.


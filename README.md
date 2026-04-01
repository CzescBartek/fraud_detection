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

## 📊 Model Evaluation Results

The model was evaluated using metrics specifically suited for highly imbalanced datasets:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **ROC AUC** | 0.98 | Excellent separation between classes. |
| **Average Precision (AP)** | 0.77 | Strong ability to catch fraud while minimizing false alarms. |

## 🏗️ Project Structure
```text
fraud_detection/
├── api/                # FastAPI application & Pydantic schemas
├── plots/              # Figures used in evaluation
├── models/             # Serialized (.pkl) models, scalers, and metadata
├── src/                # Core ML logic (Preprocessing, Training, Evaluator)
├── __main__.py         # Package entry point for the training pipeline
├── Dockerfile          # Container configuration for production
├── .dockerignore       # Files to exclude from the Docker image
└── requirements.txt    # Python dependencies with pinned versions

---
## 🔗 Let's Connect!

If you have any questions about this Fraud Detection system or want to discuss ML Engineering, feel free to reach out:

* 💼 **LinkedIn:** [www.linkedin.com/in/bartosz-pliszka-b502bb359]
* 📧 **Email:** [bartekpliszka@op.pl]

---
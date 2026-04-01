from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

class TransactionData(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float


app = FastAPI(title="Fraud Detection API", description="System wykrywania oszustw w czasie rzeczywistym")


MODEL_PATH = 'models/fraud_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    print("BŁĄD: Brak plików modelu lub scalera w folderze models/")

@app.get("/")
def home():
    return {"status": "API is running", "model": "Random Forest Fraud Detector"}

@app.post("/predict")
def predict_fraud(data: TransactionData):
    try:

        input_dict = data.dict()
        df = pd.DataFrame([input_dict])


        df['time_delta'] = 0.0
        df['feat_dist'] = 0.0
        df['amount_ratio'] = 1.0
        df['amount_zscore'] = 0.0
        feature_names = joblib.load('models/feature_names.pkl')
        df = df.reindex(columns=feature_names, fill_value=0.0)
        X_scaled = scaler.transform(df)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        return {
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(probability), 4),
            "action": "BLOCK" if prediction else "ALLOW"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
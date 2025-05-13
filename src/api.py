from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="A simple REST API to predict fraudulent transactions",
    version="1.0.0"
)

# Load model and scaler
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or Scaler not found. Please train and save them first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Input data structure
class Transaction(BaseModel):
    features: list  # 30 feature values (excluding Time), in correct order

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        input_data = np.array(transaction.features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        proba = model.predict_proba(scaled_data)[0][1]

        return {
            "prediction": int(prediction),
            "fraud_probability": round(float(proba), 4),
            "message": "Fraudulent transaction" if prediction == 1 else "Legitimate transaction"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

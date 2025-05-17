from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load best model (e.g., logistic regression)
model_path = "models/logistic_regression.joblib"
model = joblib.load(model_path)

app = FastAPI(title="Credit Card Fraud Detection API")

class Transaction(BaseModel):
    features: list  # expects a list of 28 features + scaled amount

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = np.array(transaction.features).reshape(1, -1)
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0].tolist()
        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        return {"error": str(e)}

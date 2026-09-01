import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Bank Subscription Inference Service")

try:
    artifact = joblib.load("models/best_model.pkl")
    model = artifact["model"]
    scaler = artifact["scaler"]
    expected_columns = artifact["columns"]
except FileNotFoundError:
    model, scaler, expected_columns = None, None ,None

class ClientInput(BaseModel):
    age: int
    balance: float
    duration: int
    campaign: int
    job: str = "management"
    marital: str = "married"
    education: str = "tertiary"
    housing: str = "yes"
    loan: str = "no"

@app.get("/health")
def health():
    return {"status": "live", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: ClientInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model artifact not found.")
    
    try:
        raw_df = pd.DataFrame([payload.model_dump()])
        encoded_df = pd.get_dummies(raw_df).reindex(columns=expected_columns, fill_value=0)

        scaled_features = scaler.transform(encoded_df)
        prediction = model.predict(scaled_features)[0]
        
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_features)[0][1]
            
        return {
            "subscription_prediction": int(prediction),
            "deposit_probability": round(float(probability), 4) if probability else "N/A"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
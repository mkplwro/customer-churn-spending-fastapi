from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["regression"] = joblib.load("models/regression_model.pkl")

    classification_bundle = joblib.load("models/classification_model.pkl")
    models["classification"] = classification_bundle["model"]
    models["threshold"] = classification_bundle["threshold"]

    yield

    models.clear()
app = FastAPI(title="Customer Analytics API", lifespan=lifespan)

@app.get("/")
def root():
    return {
        "message": "Customer Analytics API is running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict_spend": "/predict/spend",
            "predict_churn": "/predict/churn",
        },
    }


class Customer(BaseModel):
    age: int
    city: str
    monthly_income_pln: float
    tenure_months: int
    avg_logins_last_30d: int
    support_tickets_last_30d: int
    satisfaction_score: int
    plan: str
    auto_renew: int
    discount_pct: int
    acquisition_channel: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/spend")
def predict_spend(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    predicted_spend = models["regression"].predict(row)[0]
    return {"predicted_monthly_spend_pln": round(float(predicted_spend), 2)}

@app.post("/predict/churn")
def predict_churn(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    probability = models["classification"].predict_proba(row)[:, 1][0]
    threshold = models["threshold"]
    return {
        "churn_probability": round(float(probability), 4),
        "threshold": round(float(threshold), 2),
        "will_churn": bool(probability >= threshold),
    }

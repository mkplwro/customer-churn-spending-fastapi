from fastapi.testclient import TestClient
from main import app

CUSTOMER = {
    "age": 34,
    "city": "warszawa",
    "monthly_income_pln": 8500.0,
    "tenure_months": 18,
    "avg_logins_last_30d": 9,
    "support_tickets_last_30d": 1,
    "satisfaction_score": 7,
    "plan": "standard",
    "auto_renew": 1,
    "discount_pct": 0,
    "acquisition_channel": "organic",
}


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_spend():
    with TestClient(app) as client:
        response = client.post("/predict/spend", json=CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_monthly_spend_pln" in data
    assert isinstance(data["predicted_monthly_spend_pln"], float)

def test_predict_churn():
    with TestClient(app) as client:
        response = client.post("/predict/churn", json=CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "threshold" in data
    assert "will_churn" in data
    assert 0 <= data["churn_probability"] <= 1
    assert 0 <= data["threshold"] <= 1
    assert isinstance(data["will_churn"], bool)
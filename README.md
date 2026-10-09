# Customer Spend & Churn Prediction

End-to-end machine learning project on a subscription-customer dataset. A **regression** model predicts monthly spend, and a **classification** model predicts customer churn. Both models are served through a FastAPI app and packaged in a Docker image.

## Results

| Task | Final model | Test results | Baseline |
|---|---|---|---|
| Monthly spend (regression) | Lasso | MAE 11.43 PLN, R² 0.86 | Median prediction: MAE 32.40 PLN |
| Churn (classification) | Logistic Regression, tuned threshold | Recall 0.85, F2 0.59, ROC-AUC 0.75 | Dummy classifier: recall 0 |

**Key findings**

- Subscription plan and tenure are the strongest drivers of spend.
- Customers with genuinely zero spend cause a large part of the regression error. This is reported as a model limitation.
- Login activity, satisfaction score and tenure are the strongest predictors of churn.
- Missed churners look like loyal, satisfied, long-tenured customers. This is a blind spot to be aware of in business use.

## Data

`data/subscription_customers_dirty.csv` contains about 10,000 synthetic subscription customers (demographics, usage, plan and billing). The data has intentional quality issues (invalid values, sentinel codes, outliers, missing values), so the cleaning steps follow a realistic workflow.

## Approach

- **Cleaning:** outlier and sentinel-value detection, with the reasoning behind each decision.
- **Modeling:** scikit-learn pipelines with `ColumnTransformer` (no train/test leakage), cross-validation, and tuning with `GridSearchCV`, `RandomizedSearchCV` and Optuna.
- **Regression:** six models compared (Linear, Ridge, Lasso, Elastic Net, Decision Tree, XGBoost). The final model is chosen by CV performance, not by complexity.
- **Classification:** stratified split to handle ~81/19 class imbalance. Eight models compared using Recall and F2 instead of accuracy, and the decision threshold is optimized for F2.

## Project structure

```
├── app/
│   ├── main.py                 # FastAPI application
│   ├── predict.py              # sample prediction script
│   ├── Dockerfile
│   └── requirements_docker.txt
├── data/                       # raw CSV dataset
├── models/                     # trained models
├── notebooks/                  # regression_model.ipynb, classification_model.ipynb
├── .github/workflows/          # GitHub Actions
└── requirements_train.txt      # dependencies for the notebooks
```

## Quick start

**Notebooks**

```bash
pip install -r requirements_train.txt
```

Make sure the CSV file is in `data/`, then run the notebooks from the `notebooks/` folder.

**API with Docker**

```bash
docker build -f app/Dockerfile -t churn-spending .
docker run --rm -p 8000:8000 churn-spending
```

Interactive API docs are available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict/spend` | Predicted monthly spend (PLN) |
| POST | `/predict/churn` | Churn probability, decision threshold and churn flag |

Example request:

```bash
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{"age": 34, "city": "warszawa", "monthly_income_pln": 8500.0, "tenure_months": 18, "avg_logins_last_30d": 9, "support_tickets_last_30d": 1, "satisfaction_score": 7, "plan": "standard", "auto_renew": 1, "discount_pct": 0, "acquisition_channel": "organic"}'
```

The response contains `churn_probability`, `threshold` and `will_churn`. To run the sample script instead of the API, use `docker run --rm churn-spending python predict.py`.

## Tech stack

Python, pandas, scikit-learn, XGBoost, LightGBM, Optuna, FastAPI, Docker
# Customer Spend & Churn Prediction

End-to-end machine learning project on a subscription-customer dataset. A **regression** model predicts monthly customer spend, while a **classification** model predicts customer churn. Both trained models are served through a FastAPI application packaged in a Docker image.

## Results

| Task | Final model | Test results | Baseline |
|---|---|---|---|
| Monthly spend (regression) | Lasso | MAE 11.43 PLN, R² 0.86 | Median prediction: MAE 32.40 PLN |
| Churn (classification) | Logistic Regression with tuned threshold | Recall 0.85, F2 0.59, ROC-AUC 0.75 | Dummy classifier: recall 0 |

### Key findings

- Subscription plan and tenure are the strongest drivers of monthly spend.
- Customers with genuinely zero spend account for a substantial part of the regression error. This is documented as a model limitation.
- Login activity, satisfaction score and tenure are the strongest predictors of churn.
- Some missed churners resemble loyal, satisfied customers with long tenure. This is an important blind spot for business use.

## Data

`data/subscription_customers_dirty.csv` contains approximately 10,000 synthetic subscription customers, including demographics, usage, plan and billing information.

The dataset includes intentional data-quality issues such as invalid values, sentinel codes, outliers and missing values. The cleaning workflow addresses these issues explicitly.

## Approach

- **Data cleaning:** outlier and sentinel-value detection, with documented reasoning behind cleaning decisions.
- **Modeling:** scikit-learn pipelines using `ColumnTransformer` to prevent train/test leakage, cross-validation and hyperparameter tuning with `GridSearchCV`, `RandomizedSearchCV` and Optuna.
- **Regression:** six models compared — Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree and XGBoost. The final model is selected based on cross-validation performance.
- **Classification:** stratified splitting to account for approximately 81/19 class imbalance. Eight models are compared using Recall and F2 rather than accuracy, and the decision threshold is tuned for F2.

## Project structure

```text
├── app/
│   ├── main.py                 # FastAPI application and prediction endpoints
│   ├── predict.py              # Sample prediction script
│   ├── Dockerfile              # Docker image definition
│   └── requirements_docker.txt # Pinned API dependencies
├── data/                       # Raw CSV dataset
├── models/
│   ├── regression_model.pkl    # Trained regression model
│   └── classification_model.pkl # Model and tuned classification threshold
├── notebooks/
│   ├── regression_model.ipynb
│   └── classification_model.ipynb
├── .github/workflows/          # GitHub Actions
├── requirements_train.txt      # Pinned training and notebook dependencies
└── README.md
```

## Environment and reproducibility

The training environment and Docker serving environment use pinned versions of the main libraries to reduce dependency drift and support consistent model loading.

The current verified environment uses:

| Component | Version |
|---|---|
| Python | 3.11.9 locally; Python 3.11 slim in Docker |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.1 |
| joblib | 1.4.2 |
| XGBoost | 3.2.0 |
| LightGBM | 4.7.0 |
| Optuna | 4.9.0 |

The Docker dependency file contains the pinned serving dependencies. The training dependency file additionally includes the libraries needed for notebooks and experimentation.

The model files were successfully loaded with scikit-learn 1.5.1, without an `InconsistentVersionWarning`. The Docker image was built successfully, and the API was started and tested locally.

For stronger long-term reproducibility, a shared constraints file or a complete lockfile covering transitive dependencies can be added.

## Quick start

### 1. Training and notebooks

Create and activate a virtual environment, then install the training dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_train.txt
```

Make sure the CSV file is available at `data/subscription_customers_dirty.csv`, then open and run the notebooks from the `notebooks/` directory.

### 2. Run the API with Docker

Build the image from the project root:

```bash
docker build --no-cache -f app/Dockerfile -t customer-churn-api .
```

Start the container:

```bash
docker run --rm -d --name customer-churn-api-test -p 8000:8000 customer-churn-api
```

Open the interactive API documentation:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Root endpoint: [http://localhost:8000/](http://localhost:8000/)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

To inspect container logs:

```bash
docker logs customer-churn-api-test
```

To stop the container:

```bash
docker stop customer-churn-api-test
```

### 3. API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information and available endpoints |
| GET | `/health` | Health check |
| POST | `/predict/spend` | Predicted monthly spend in PLN |
| POST | `/predict/churn` | Churn probability, decision threshold and churn flag |

Both prediction endpoints accept a JSON request with the following fields:

| Field | Type | Example |
|---|---|---|
| `age` | integer | `34` |
| `city` | string | `"warszawa"` |
| `monthly_income_pln` | float | `8500.0` |
| `tenure_months` | integer | `18` |
| `avg_logins_last_30d` | integer | `9` |
| `support_tickets_last_30d` | integer | `1` |
| `satisfaction_score` | integer | `7` |
| `plan` | string | `"standard"` |
| `auto_renew` | integer | `1` |
| `discount_pct` | integer | `0` |
| `acquisition_channel` | string | `"organic"` |

### 4. Example request: churn prediction

```bash
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
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
    "acquisition_channel": "organic"
  }'
```

The response includes:

- `churn_probability` — predicted probability of churn.
- `threshold` — tuned decision threshold used by the classifier.
- `will_churn` — Boolean indicating whether the probability meets or exceeds the threshold.

### 5. Example request: spend prediction

Use the same JSON customer payload with:

```text
POST http://localhost:8000/predict/spend
```

The response contains `predicted_monthly_spend_pln`.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Optuna, joblib, FastAPI, Pydantic, Uvicorn and Docker.
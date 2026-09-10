import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# -----------------------------
# Load processed dataset
# -----------------------------
DATA_PATH = "data/processed/hour_processed.csv"

df = pd.read_csv(DATA_PATH)


print(f"Dataset loaded successfully: {df.shape}")


# -----------------------------
# Features and Target
# -----------------------------
target = "cnt"

drop_columns = [
    "instant",
    "dteday",
    "casual",
    "registered",
    "cnt"
]

X = df.drop(columns=drop_columns)
y = df[target]


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    rmse = (mean_squared_error(y_test, predictions)) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4)
    }


# -----------------------------
# Baseline Model
# -----------------------------
print("\nTraining Baseline Model (Linear Regression)...")

baseline = LinearRegression()

baseline.fit(X_train, y_train)

baseline_metrics = evaluate(
    baseline,
    X_test,
    y_test
)

print("Baseline Metrics")
print(baseline_metrics)


# -----------------------------
# Candidate Model
# -----------------------------
print("\nTraining Candidate Model (Random Forest)...")

candidate = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

candidate.fit(
    X_train,
    y_train
)

candidate_metrics = evaluate(
    candidate,
    X_test,
    y_test
)

print("Candidate Metrics")
print(candidate_metrics)


# -----------------------------
# Model Promotion Rule
# -----------------------------
if candidate_metrics["RMSE"] < baseline_metrics["RMSE"]:

    best_model = candidate
    best_metrics = candidate_metrics
    model_name = "RandomForest"

else:

    best_model = baseline
    best_metrics = baseline_metrics
    model_name = "LinearRegression"


print(f"\nSelected Model : {model_name}")


# -----------------------------
# Save Model
# -----------------------------
Path("models").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "models/model.pkl"
)

print("Model saved successfully.")


# -----------------------------
# Save Evaluation Report
# -----------------------------
Path("artifacts/eval").mkdir(
    parents=True,
    exist_ok=True
)

report = {
    "baseline": baseline_metrics,
    "candidate": candidate_metrics,
    "selected_model": model_name
}

with open(
    "artifacts/eval/evaluation.json",
    "w"
) as f:
    json.dump(
        report,
        f,
        indent=4
    )

print("Evaluation report saved.")
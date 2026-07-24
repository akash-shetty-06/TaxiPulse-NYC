import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
)
MODEL_FILE = PROJECT_ROOT / "models" / "taxi_demand_model.joblib"
METRICS_FILE = PROJECT_ROOT / "reports" / "metrics.json"
PREDICTIONS_FILE = PROJECT_ROOT / "reports" / "predictions.parquet"
IMPORTANCE_FILE = PROJECT_ROOT / "reports" / "feature_importance.csv"

TEST_START_DATE = pd.Timestamp("2025-01-25")

FEATURE_COLUMNS = [
    "PULocationID",
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "lag_1_hour",
    "lag_24_hours",
    "lag_168_hours",
    "rolling_mean_24_hours",
    "rolling_mean_168_hours",
]

TARGET_COLUMN = "trip_count"


def weighted_absolute_percentage_error(
    actual: pd.Series,
    predicted: np.ndarray,
) -> float:
    """Calculate WAPE while safely handling zero-demand records."""

    absolute_error = np.abs(actual.to_numpy() - predicted)
    denominator = np.abs(actual.to_numpy()).sum()

    if denominator == 0:
        return 0.0

    return float(absolute_error.sum() / denominator)


def evaluate(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate regression evaluation metrics."""

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "wape": weighted_absolute_percentage_error(actual, predicted),
    }


def main() -> None:
    """Train and evaluate the taxi-demand forecasting model."""

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model features...")
    data = pd.read_parquet(FEATURE_FILE)

    train_data = data[data["pickup_hour"] < TEST_START_DATE].copy()
    test_data = data[data["pickup_hour"] >= TEST_START_DATE].copy()

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    print(f"Training rows: {len(train_data):,}")
    print(f"Testing rows: {len(test_data):,}")

    baseline_predictions = test_data["lag_168_hours"].to_numpy()
    baseline_metrics = evaluate(y_test, baseline_predictions)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Training XGBoost model...")
    model.fit(x_train, y_train)

    model_predictions = np.clip(model.predict(x_test), 0, None)
    model_metrics = evaluate(y_test, model_predictions)

    metrics = {
        "test_start_date": str(TEST_START_DATE.date()),
        "training_rows": len(train_data),
        "testing_rows": len(test_data),
        "baseline": baseline_metrics,
        "xgboost": model_metrics,
    }

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "test_start_date": TEST_START_DATE,
        },
        MODEL_FILE,
    )

    prediction_results = test_data[
        ["pickup_hour", "PULocationID", TARGET_COLUMN]
    ].copy()

    prediction_results["baseline_prediction"] = baseline_predictions
    prediction_results["model_prediction"] = model_predictions
    prediction_results.to_parquet(PREDICTIONS_FILE, index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importance.to_csv(IMPORTANCE_FILE, index=False)

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nBaseline metrics:")
    print(json.dumps(baseline_metrics, indent=2))

    print("\nXGBoost metrics:")
    print(json.dumps(model_metrics, indent=2))

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "hourly_taxi_demand.parquet"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create time and historical-demand features."""

    data = data.sort_values(
        ["PULocationID", "pickup_hour"]
    ).copy()

    data["hour"] = data["pickup_hour"].dt.hour
    data["day_of_week"] = data["pickup_hour"].dt.dayofweek
    data["day_of_month"] = data["pickup_hour"].dt.day
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype("int8")

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )
    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )
    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    zone_demand = data.groupby(
        "PULocationID",
        observed=True,
    )["trip_count"]

    data["lag_1_hour"] = zone_demand.shift(1)
    data["lag_24_hours"] = zone_demand.shift(24)
    data["lag_168_hours"] = zone_demand.shift(168)

    data["rolling_mean_24_hours"] = zone_demand.transform(
        lambda values: values.shift(1).rolling(24).mean()
    )

    data["rolling_mean_168_hours"] = zone_demand.transform(
        lambda values: values.shift(1).rolling(168).mean()
    )

    data = data.dropna().reset_index(drop=True)

    return data


def main() -> None:
    """Load processed demand data and save model features."""

    print("Loading hourly taxi demand...")
    demand = pd.read_parquet(INPUT_FILE)

    print("Creating forecasting features...")
    features = create_features(demand)

    features.to_parquet(OUTPUT_FILE, index=False)

    print(f"Feature rows: {len(features):,}")
    print(f"Feature columns: {len(features.columns)}")
    print(f"Features saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
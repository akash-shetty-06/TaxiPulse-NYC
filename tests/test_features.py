import pandas as pd

from src.features import create_features


def create_sample_data() -> pd.DataFrame:
    """Create predictable hourly demand for one taxi zone."""

    return pd.DataFrame(
        {
            "pickup_hour": pd.date_range(
                start="2025-01-01",
                periods=200,
                freq="h",
            ),
            "PULocationID": [100] * 200,
            "trip_count": list(range(200)),
        }
    )


def test_create_features_adds_expected_columns() -> None:
    """Feature engineering should create all model inputs."""

    features = create_features(create_sample_data())

    expected_columns = {
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "lag_1_hour",
        "lag_24_hours",
        "lag_168_hours",
        "rolling_mean_24_hours",
        "rolling_mean_168_hours",
    }

    assert expected_columns.issubset(features.columns)
    assert not features.isna().any().any()


def test_lag_features_use_only_past_demand() -> None:
    """Lag values must come from earlier hours, not the future."""

    features = create_features(create_sample_data())
    first_row = features.iloc[0]

    assert first_row["trip_count"] == 168
    assert first_row["lag_1_hour"] == 167
    assert first_row["lag_24_hours"] == 144
    assert first_row["lag_168_hours"] == 0


def test_rolling_mean_uses_previous_24_hours() -> None:
    """The rolling mean must exclude the current hour."""

    features = create_features(create_sample_data())
    first_row = features.iloc[0]

    expected_mean = sum(range(144, 168)) / 24

    assert first_row["rolling_mean_24_hours"] == expected_mean
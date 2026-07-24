import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_FILE = PROJECT_ROOT / "reports" / "metrics.json"
PREDICTIONS_FILE = PROJECT_ROOT / "reports" / "predictions.parquet"
IMPORTANCE_FILE = PROJECT_ROOT / "reports" / "feature_importance.csv"


st.set_page_config(
    page_title="TaxiPulse NYC",
    page_icon=None,
    layout="wide",
)


@st.cache_data
def load_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load saved model results."""

    with METRICS_FILE.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    predictions = pd.read_parquet(PREDICTIONS_FILE)
    importance = pd.read_csv(IMPORTANCE_FILE)

    return metrics, predictions, importance


metrics, predictions, importance = load_data()

st.title("TaxiPulse NYC")
st.write(
    "Hourly NYC Yellow Taxi demand forecasting with an "
    "XGBoost model and time-based evaluation."
)

baseline = metrics["baseline"]
model = metrics["xgboost"]

mae_improvement = (
    (baseline["mae"] - model["mae"]) / baseline["mae"] * 100
)

column_1, column_2, column_3, column_4 = st.columns(4)

column_1.metric(
    "Model MAE",
    f"{model['mae']:.2f} trips",
)

column_2.metric(
    "Model RMSE",
    f"{model['rmse']:.2f} trips",
)

column_3.metric(
    "Model WAPE",
    f"{model['wape']:.1%}",
)

column_4.metric(
    "MAE Improvement",
    f"{mae_improvement:.1f}%",
)

st.divider()

available_zones = sorted(predictions["PULocationID"].unique())

selected_zone = st.sidebar.selectbox(
    "Select pickup zone",
    options=["All zones"] + available_zones,
)

if selected_zone == "All zones":
    chart_data = (
        predictions.groupby("pickup_hour", as_index=False)
        .agg(
            actual_demand=("trip_count", "sum"),
            predicted_demand=("model_prediction", "sum"),
            baseline_demand=("baseline_prediction", "sum"),
        )
    )
    chart_title = "Total Hourly Taxi Demand Across All Zones"
else:
    chart_data = predictions[
        predictions["PULocationID"] == selected_zone
    ].copy()

    chart_data = chart_data.rename(
        columns={
            "trip_count": "actual_demand",
            "model_prediction": "predicted_demand",
            "baseline_prediction": "baseline_demand",
        }
    )

    chart_title = f"Hourly Taxi Demand for Pickup Zone {selected_zone}"

line_data = chart_data.melt(
    id_vars="pickup_hour",
    value_vars=[
        "actual_demand",
        "predicted_demand",
        "baseline_demand",
    ],
    var_name="series",
    value_name="trip_count",
)

line_figure = px.line(
    line_data,
    x="pickup_hour",
    y="trip_count",
    color="series",
    title=chart_title,
    labels={
        "pickup_hour": "Pickup hour",
        "trip_count": "Number of trips",
        "series": "Series",
    },
)

line_figure.update_layout(
    hovermode="x unified",
    legend_title_text="",
)

st.plotly_chart(line_figure, use_container_width=True)

st.subheader("Model Comparison")

comparison = pd.DataFrame(
    {
        "Model": ["Weekly baseline", "XGBoost"],
        "MAE": [baseline["mae"], model["mae"]],
        "RMSE": [baseline["rmse"], model["rmse"]],
        "WAPE": [baseline["wape"], model["wape"]],
    }
)

st.dataframe(
    comparison.style.format(
        {
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "WAPE": "{:.1%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Feature Importance")

    importance_figure = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        labels={
            "importance": "Importance",
            "feature": "Feature",
        },
    )

    importance_figure.update_layout(showlegend=False)
    st.plotly_chart(importance_figure, use_container_width=True)

with right_column:
    st.subheader("Zones with the Largest Errors")

    zone_errors = predictions.copy()
    zone_errors["absolute_error"] = (
        zone_errors["trip_count"] - zone_errors["model_prediction"]
    ).abs()

    zone_summary = (
        zone_errors.groupby("PULocationID", as_index=False)
        .agg(
            average_actual_demand=("trip_count", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .sort_values("mean_absolute_error", ascending=False)
        .head(15)
    )

    st.dataframe(
        zone_summary.style.format(
            {
                "average_actual_demand": "{:.2f}",
                "mean_absolute_error": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Evaluation period: January 25–31, 2025. "
    "The baseline uses demand from the same hour one week earlier."
)
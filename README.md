# TaxiPulse NYC

TaxiPulse NYC is an end-to-end data science project that forecasts hourly NYC Yellow Taxi demand for individual pickup zones.

The project includes data acquisition, cleaning, time-series feature engineering, baseline evaluation, XGBoost modeling, automated tests, and an interactive Streamlit dashboard.

## Results

The model was evaluated on the final seven days of January 2025 using a time-based train-test split.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Weekly baseline | 4.82 | 15.86 | 25.2% |
| XGBoost | 3.93 | 13.01 | 20.5% |

XGBoost reduced mean absolute error by approximately **18.6%** compared with using demand from the same hour one week earlier.

## Dashboard

The Streamlit dashboard includes:

- Actual, predicted, and baseline hourly demand
- Individual pickup-zone selection
- Model evaluation metrics
- Feature-importance visualization
- Zones with the largest forecasting errors

A public dashboard link will be added after deployment.

## Dataset

The project uses official January 2025 NYC Yellow Taxi trip records published by the New York City Taxi and Limousine Commission.

Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

The raw dataset contains:

- 3,475,226 taxi trips
- Pickup and drop-off timestamps
- Pickup and drop-off zone identifiers
- Trip distance
- Fare and payment information

After validation, 3,465,683 trips were aggregated into 192,696 hourly pickup-zone records.

Raw and processed datasets are excluded from Git because they can be reproduced with the included pipeline.

## Methodology

### 1. Data acquisition

`src/data_pipeline.py` downloads the official Parquet dataset in chunks and avoids downloading an existing file again.

### 2. Data preparation

The pipeline:

- Removes missing pickup timestamps and zone identifiers
- Keeps trips from January 2025
- Keeps valid taxi-zone identifiers
- Aggregates pickups by hour and pickup zone
- Adds missing zone-hour combinations with zero demand
- Saves the processed data in Parquet format

### 3. Feature engineering

`src/features.py` creates:

- Hour of day
- Day of week
- Day of month
- Weekend indicator
- Cyclical hour and weekday features
- One-hour demand lag
- Twenty-four-hour demand lag
- One-week demand lag
- Twenty-four-hour rolling demand average
- One-week rolling demand average

All lag and rolling features use only earlier demand to prevent future-data leakage.

### 4. Model training

`src/train.py` compares:

- Weekly persistence baseline
- XGBoost regression model

Data before January 25 is used for training. January 25–31 is reserved for testing.

Evaluation metrics include:

- Mean Absolute Error
- Root Mean Squared Error
- Weighted Absolute Percentage Error

### 5. Testing

Automated Pytest tests verify that:

- Expected model features are created
- Feature data contains no missing values
- Lag features use previous observations
- Rolling averages exclude the current hour

## Project Structure

```text
TaxiPulse-NYC/
├── dashboard/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── reports/
│   ├── feature_importance.csv
│   ├── metrics.json
│   └── predictions.parquet
├── src/
│   ├── data_pipeline.py
│   ├── features.py
│   └── train.py
├── tests/
│   └── test_features.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt

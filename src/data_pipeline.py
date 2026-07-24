from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "trip-data/yellow_tripdata_2025-01.parquet"
)

RAW_DATA_FILE = RAW_DATA_DIR / "yellow_tripdata_2025-01.parquet"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "hourly_taxi_demand.parquet"


def download_dataset() -> Path:
    """Download one month of NYC Yellow Taxi trip data."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_FILE.exists():
        print(f"Dataset already exists: {RAW_DATA_FILE}")
        return RAW_DATA_FILE

    print("Downloading NYC Yellow Taxi data...")

    with requests.get(DATA_URL, stream=True, timeout=60) as response:
        response.raise_for_status()

        with RAW_DATA_FILE.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print(f"Dataset saved to: {RAW_DATA_FILE}")
    return RAW_DATA_FILE


def build_hourly_demand(raw_file: Path) -> Path:
    """Clean trip records and aggregate pickups by zone and hour."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading raw taxi data...")

    trips = pd.read_parquet(
        raw_file,
        columns=["tpep_pickup_datetime", "PULocationID"],
    )

    initial_rows = len(trips)

    trips = trips.dropna(
        subset=["tpep_pickup_datetime", "PULocationID"]
    ).copy()

    trips = trips[
        trips["tpep_pickup_datetime"].between(
            "2025-01-01",
            "2025-02-01",
            inclusive="left",
        )
    ]

    trips = trips[
        trips["PULocationID"].between(1, 263)
    ].copy()

    trips["PULocationID"] = trips["PULocationID"].astype("int16")
    trips["pickup_hour"] = trips["tpep_pickup_datetime"].dt.floor("h")

    hourly_demand = (
        trips.groupby(
            ["pickup_hour", "PULocationID"],
            observed=True,
        )
        .size()
        .rename("trip_count")
    )

    all_hours = pd.date_range(
        start="2025-01-01",
        end="2025-02-01",
        freq="h",
        inclusive="left",
    )

    all_zones = sorted(trips["PULocationID"].unique())

    complete_index = pd.MultiIndex.from_product(
        [all_hours, all_zones],
        names=["pickup_hour", "PULocationID"],
    )

    hourly_demand = (
        hourly_demand.reindex(complete_index, fill_value=0)
        .reset_index()
        .sort_values(["pickup_hour", "PULocationID"])
    )

    hourly_demand["trip_count"] = hourly_demand["trip_count"].astype(
        "int32"
    )

    hourly_demand.to_parquet(PROCESSED_DATA_FILE, index=False)

    print(f"Raw rows: {initial_rows:,}")
    print(f"Valid rows: {len(trips):,}")
    print(f"Hourly records: {len(hourly_demand):,}")
    print(f"Processed data saved to: {PROCESSED_DATA_FILE}")

    return PROCESSED_DATA_FILE


def main() -> None:
    """Run the complete data preparation pipeline."""

    raw_file = download_dataset()
    build_hourly_demand(raw_file)


if __name__ == "__main__":
    main()
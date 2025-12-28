import pandas as pd
import numpy as np
import re
from typing import List


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop Rows with Missing Target Variable (SolarGeneration)
    initial_rows = len(df)
    df = df.dropna(subset=["SolarGeneration"])
    print(f"Dropped {initial_rows - len(df)} rows with missing Target.")

    # 2. Handle Missing Weather Data (Linear Interpolation)
    weather_cols = [
        "ApparentTemperature",
        "AirTemperature",
        "DewPointTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "WindDirection",
    ]

    existing_weather_cols = [c for c in weather_cols if c in df.columns]
    df.loc[:, existing_weather_cols] = df.loc[:, existing_weather_cols].interpolate(
        method="linear", limit_direction="both"
    )

    # 3. Clip Negative Solar Generation Values to Zero
    df.loc[:, "SolarGeneration"] = df["SolarGeneration"].clip(lower=0)

    # 4. Change Null Optimizer Values to 'None'
    if "Optimizers" in df.columns:
        df.loc[:, "Optimizers"] = df["Optimizers"].fillna("None")

    # 5. Number of Panels Cleanup
    if "Number of panels" in df.columns:
        df.rename(columns={"Number of panels": "NumberOfPanels"}, inplace=True)

    return df


def filter_valid_sites(df: pd.DataFrame) -> pd.DataFrame:
    # Remove Sites with Missing kWp Capacity
    missing_kwp = df["kWp"].isnull().sum()
    if missing_kwp > 0:
        print(f"\nDropping {missing_kwp} rows with missing 'kWp' capacity data.")
        df = df.dropna(subset=["kWp"])

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Format Timestamp to Datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Extract Standard Time Features
    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear
    df["Year"] = df["Timestamp"].dt.year

    # Day / Night Indicator
    df["IsDaylight"] = (df["Hour"] >= 6) & (df["Hour"] <= 20)

    # Encode Cyclical Features
    df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df


def get_distinct_values(df: pd.DataFrame, columns: List[str]):
    for column in columns:
        if column in df.columns:
            distinct_values = df[column].dropna().unique().tolist()
            print(f"Distinct values in '{column}': {distinct_values}")
        else:
            print(f"Column '{column}' does not exist in the DataFrame.")


def extract_hardware_specs(df):
    df = df.copy()

    # 1. Panels: Extract Wattage Rating
    def get_panel_watts(name):
        if pd.isna(name) or name == "TBD":
            return 330.0
        if "435" in str(name):
            return 435.0
        match = re.search(r"(\d+)W", str(name))
        return float(match.group(1)) if match else 330.0

    df["PanelRatingW"] = df["Panel"].apply(get_panel_watts)

    # 2. Inverters: Total Capacity in kW
    def get_inv_cap(name):
        if pd.isna(name):
            return 0.0
        name_str = str(name)

        # Handle Multiple Inverters
        matches = re.findall(r"(\d+)\s*[xX]\s*.*?(\d+\.?\d*)K?", name_str)
        total = sum(float(count) * float(rating) for count, rating in matches)
        return total

    df["TotalInverterKW"] = df["Inverter"].apply(get_inv_cap)

    # 3. Optimizers: Count
    df["OptimizerCount"] = (
        df["Optimizers"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .fillna(0)
    )

    print(f"\nExtracted hardware specs: PanelRatingW, TotalInverterKW, OptimizerCount")

    return df

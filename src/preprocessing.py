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


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Format Timestamp to Datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # 2. Extract Standard Time Features
    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear
    df["Year"] = df["Timestamp"].dt.year

    # 3. Day / Night Indicator
    df["IsDaylight"] = (df["Hour"] >= 6) & (df["Hour"] <= 20)

    # 4. Encode Cyclical Features
    df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # 5. Solar Elevation Feature
    if "lat" in df.columns and "Lon" in df.columns:
        df["SolarElevation"] = calculate_solar_elevation(
            df["lat"], df["Lon"], df["Timestamp"]
        )

        df["SolarElevation"] = df["SolarElevation"].clip(lower=0)

    # 6. Temperature-Based Features
    if "AirTemperature" in df.columns:
        # Temperature Deviation from Optimal (25°C)
        df["TempDeviation"] = (df["AirTemperature"] - 25).abs()

        # High Temperature Penalty (above 25°C)
        df["HighTempPenalty"] = (df["AirTemperature"] - 25).clip(lower=0)

    if "DewPointTemperature" in df.columns and "AirTemperature" in df.columns:
        # Temperature Dew Spread
        df["TempDewSpread"] = df["AirTemperature"] - df["DewPointTemperature"]

    # 7. Weather Interaction Features
    if "RelativeHumidity" in df.columns and "AirTemperature" in df.columns:
        # Hight Humidity & Temperature Interaction (Cloud / Haze Effect)
        df["HumidityTemp"] = df["RelativeHumidity"] * df["AirTemperature"] / 100

    if "WindSpeed" in df.columns and "AirTemperature" in df.columns:
        # Wind Speed & Temperature Interaction (Cooling Effect)
        df["WindCoolingEffect"] = df["WindSpeed"] * df["AirTemperature"]

    # 8. System Efficiency Ratios
    if "kWp" in df.columns and "NumberOfPanels" in df.columns:
        # Average Panel kW Rating
        df["AvgPanelKW"] = df["kWp"] / (df["NumberOfPanels"] + 0.001)

    if "TotalInverterKW" in df.columns and "kWp" in df.columns:
        # Inverter to Panel Capacity Ratio
        df["InverterPanelRatio"] = df["TotalInverterKW"] / (df["kWp"] + 0.001)

    if "OptimizerCount" in df.columns and "NumberOfPanels" in df.columns:
        # Optimizer Coverage (Percentage of Panels with Optimizers)
        df["OptimizerCoverage"] = df["OptimizerCount"] / (df["NumberOfPanels"] + 0.001)

    # 9. Seasonal Features
    df["Season"] = df["Month"].apply(
        lambda x: (
            0
            if x in [12, 1, 2]  # Summer
            else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3  # Autumn  # Winter
        )  # Spring
    )

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    initial_rows = len(df)

    # 1. Remove Nighttime Generation Values (SolarElevation <= 0)
    if "SolarElevation" in df.columns:
        night_mask = df["SolarElevation"] <= 0
        df.loc[night_mask, "SolarGeneration"] = df.loc[
            night_mask, "SolarGeneration"
        ].clip(upper=0.01)
        print(f"Capped {night_mask.sum()} nighttime generation values")

    # 2. Remove Physically Impossible Generation Values (Beyond 120% of kWp)
    if "kWp" in df.columns:
        theoretical_max = df["kWp"] * 1.2
        impossible_mask = df["SolarGeneration"] > theoretical_max
        impossible_count = impossible_mask.sum()

        if impossible_count > 0:
            print(
                f"Removing {impossible_count} rows with physically impossible generation (>{theoretical_max.max():.2f} kWh)"
            )
            df = df[~impossible_mask]

    # 3. Remove Statistical Outliers Using IQR Method (Per Site)
    if "SiteKey" in df.columns:

        def remove_site_outliers(group):
            # Only Check Daylight Data
            daylight = (
                group[group["SolarElevation"] > 0]
                if "SolarElevation" in group.columns
                else group
            )

            if len(daylight) < 10:
                return group

            Q1 = daylight["SolarGeneration"].quantile(0.25)
            Q3 = daylight["SolarGeneration"].quantile(0.75)
            IQR = Q3 - Q1

            # Remove Outliers Beyond 3 * IQR
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outlier_mask = (group["SolarGeneration"] < lower_bound) | (
                group["SolarGeneration"] > upper_bound
            )
            return group[~outlier_mask]

        df = df.groupby("SiteKey", group_keys=False).apply(remove_site_outliers, include_groups=False)
        df = df.reset_index(drop=True)

    rows_removed = initial_rows - len(df)
    print(
        f"Total outliers removed: {rows_removed} ({rows_removed/initial_rows*100:.2f}%)"
    )

    return df


def calculate_solar_elevation(lat, lon, timestamp):
    lat = np.array(lat)
    lon = np.array(lon)

    # 1. Extract Time Components
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    day_of_year = timestamp.dt.dayofyear

    # 2. Calculate Solar Elevation Angle
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))

    # 3. Hour Angle
    hour_angle = 15 * (hour - 12)

    # 4. Solar Elevation Calculation
    lat_rad = np.radians(lat)
    dec_rad = np.radians(declination)
    hour_rad = np.radians(hour_angle)

    elevation = np.degrees(
        np.arcsin(
            np.sin(lat_rad) * np.sin(dec_rad)
            + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
        )
    )

    return elevation


def get_distinct_values(df: pd.DataFrame, columns: List[str]):
    for column in columns:
        if column in df.columns:
            distinct_values = df[column].dropna().unique().tolist()
            print(f"Distinct values in '{column}': {distinct_values}")
        else:
            print(f"Column '{column}' does not exist in the DataFrame.")

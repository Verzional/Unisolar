import pandas as pd
from typing import List


def load_data(file_paths: List[str]):
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
            print(f"Loaded {path} | Shape: {df.shape}")
        except FileNotFoundError:
            print(f"File not found at {path}")
    return dataframes


def merge_df(
    df_generation: pd.DataFrame, df_weather: pd.DataFrame, df_details: pd.DataFrame
):
    """
    Performs a high-granularity merge (timestamp level) using compound keys
    to prevent Cartesian Joins and MemoryError.

    Args:
        df_generation (Base): Solar_Energy_Generation.csv
        df_weather: Weather_Data_reordered_all.csv
        df_details: Solar_Site_Details.csv

    Returns: The single merged pandas DataFrame.
    """

    # Ensure Timestamp is in datetime format for better merging and future use
    # Note: This is a crucial preprocessing step often needed before merging on time
    df_generation["Timestamp"] = pd.to_datetime(df_generation["Timestamp"])
    df_weather["Timestamp"] = pd.to_datetime(df_weather["Timestamp"])

    # --- Step 1: Merge Weather Data (Compound Key: CampusKey + Timestamp) ---
    print(f"Merge Starting. Base Shape: {df_generation.shape}")
    merged_df = pd.merge(
        left=df_generation, right=df_weather, on=["CampusKey", "Timestamp"], how="left"
    )
    print(f"After Merging Weather Data: {merged_df.shape}")

    # --- Step 2: Merge Site Details (Compound Key: CampusKey + SiteKey) ---
    # Note: df_details is small, so this should be fast.
    merged_df = pd.merge(
        left=merged_df,
        right=df_details,
        on=["CampusKey", "SiteKey"],
        how="left",
        suffixes=("_base", "_site_details"),
    )
    print(f"After Merging Site Details: {merged_df.shape}")

    # Final cleanup (optional)
    merged_df = merged_df.reset_index(drop=True)

    return merged_df

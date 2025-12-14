import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import joblib


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float
):
    """
    Splits data into X (features) and y (target).
    """
    feature_cols = [
        "AirTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "WindDirection",
        "kWp",
        "lat",
        "Lon",
        "Hour",
        "Month",
        "DayOfYear",
        "IsDaylight",
    ]

    X = df[feature_cols]
    y = df[target_col]

    # Simple random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a LightGBM Regressor.
    """
    print("Initializing LightGBM...")
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,  # -1 means no limit (LightGBM grows leaf-wise)
        num_leaves=31,  # Controls complexity
        random_state=42,
        n_jobs=-1,
    )

    print("Training Model...")
    model.fit(X_train, y_train)
    print("Training Complete.")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Prints evaluation metrics.
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"--- Model Performance ---")
    print(f"MAE (Mean Absolute Error): {mae:.4f} kWh")
    print(f"RMSE (Root Mean Sq Error): {rmse:.4f} kWh")
    print(f"R2 Score: {r2:.4f}")

    return predictions


def save_model(model, filepath: str):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

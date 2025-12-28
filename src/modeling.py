from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib


def split_data(df: pd.DataFrame, target_col: str, test_size: float):
    # Sort by Timestamp
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    # Define Features and Target
    feature_cols = [
        "AirTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "WindDirection",
        "kWp",
        "Number of panels",
        "PanelRatingW",
        "TotalInverterKW",
        "OptimizerCount",
        "Hour",
        "Month",
        "DayOfYear",
        "IsDaylight",
    ]

    X = df[feature_cols]
    y = df[target_col]

    # Calculate Split Index
    split_index = int(len(df) * (1 - test_size))

    # Train Test Split
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Define Hyperparameter Grid
    param_dist = {
        "num_leaves": [31, 50, 70],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [1000, 2000],
        "max_depth": [-1, 10, 15],
    }

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=3)

    model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=5,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"Best Params: {search.best_params_}")
    print("\nTraining Complete.")
    
    return search.best_estimator_

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"MAE (Mean Absolute Error): {mae:.4f} kWh")
    print(f"RMSE (Root Mean Sq Error): {rmse:.4f} kWh")
    print(f"R2 Score: {r2:.4f}")

    return predictions


def save_model(model, filepath: str):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

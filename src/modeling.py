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
        # Weather Features
        "AirTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "WindDirection",
        
        # System Specifications
        "kWp",
        "NumberOfPanels",
        "PanelRatingW",
        "TotalInverterKW",
        "OptimizerCount",
        
        # Time Features
        "Hour",
        "Month",
        "DayOfYear",
        "IsDaylight",
        "HourSin",
        "HourCos",
        "MonthSin",
        "MonthCos",
        "Season",
        
        # Solar Position Features
        "SolarElevation",
        
        # Temperature-Based Features
        "TempDeviation",
        "HighTempPenalty",
        "TempDewSpread",
        
        # Weather Interaction Features
        "HumidityTemp",
        "WindCoolingEffect",
        
        # System Efficiency Ratios
        "AvgPanelKW",
        "InverterPanelRatio",
        "OptimizerCoverage",
    ]
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df[target_col]

    # Calculate Split Index
    split_index = int(len(df) * (1 - test_size))

    # Train Test Split
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Hyperparameter Grid
    param_dist = {
        # Tree Structure
        "num_leaves": [31, 63, 100],
        "max_depth": [10, 15, 20],
        "min_child_samples": [20, 50, 100],
        
        # Learning
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [500, 1000, 1500], 
        
        # Sampling 
        "subsample": [0.8, 0.9, 1.0],
        "subsample_freq": [1],
        "colsample_bytree": [0.8, 0.9, 1.0],
        
        # Regularization 
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [0, 0.1, 1],
    }

    # CV Splitter
    tscv = TimeSeriesSplit(n_splits=3)

    # Model Initialization
    model = LGBMRegressor(
        random_state=42, 
        n_jobs=-1, 
        verbose=-1,
        force_col_wise=True,
        importance_type='gain',
        boosting_type='gbdt',  
    )

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=25,  
        scoring="r2", 
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        error_score='raise',  
    )

    search.fit(X_train, y_train)

    print(f"\nBest Params: {search.best_params_}")
    print(f"Best CV R2 Score: {search.best_score_:.4f}")
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

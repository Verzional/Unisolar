import os
import sys
import joblib
import pandas as pd
from flask_cors import CORS
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
from dotenv import load_dotenv

from functions import (
    get_weather_data,
    get_weather_forecast,
    geocode_location,
    reverse_geocode_coords,
    prepare_features,
    estimate_generation_simple,
    get_location_from_request,
    get_system_config_from_request,
    make_prediction,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import calculate_solar_elevation

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "solar_model.pkl")

# Load Model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "openweather_configured": bool(OPENWEATHER_API_KEY),
    })


@app.route("/api/geocode", methods=["GET"])
def geocode():
    """Geocode a location name to coordinates."""
    location = request.args.get("location", "")
    if not location:
        return jsonify({"error": "Location parameter is required"}), 400

    result, error = geocode_location(location)
    if error:
        return jsonify({"error": error}), 400

    return jsonify(result)


@app.route("/api/reverse-geocode", methods=["GET"])
def reverse_geocode():
    """Reverse geocode coordinates to location name."""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    result, error = reverse_geocode_coords(lat, lon)
    if error:
        return jsonify({"error": error}), 400

    return jsonify(result)


@app.route("/api/weather", methods=["GET"])
def get_weather():
    """Get current weather for a location."""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    location = request.args.get("location", "")

    # Geocode if location name provided
    if location and (lat is None or lon is None):
        geo_result, error = geocode_location(location)
        if error:
            return jsonify({"error": error}), 400
        lat, lon = geo_result["lat"], geo_result["lon"]

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    weather, error = get_weather_data(lat, lon)
    if error:
        return jsonify({"error": error}), 400

    return jsonify({"weather": weather, "location": {"lat": lat, "lon": lon}})


@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    """Get weather forecast for a location."""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    location = request.args.get("location", "")

    # Geocode if location name provided
    if location and (lat is None or lon is None):
        geo_result, error = geocode_location(location)
        if error:
            return jsonify({"error": error}), 400
        lat, lon = geo_result["lat"], geo_result["lon"]

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    forecasts, error = get_weather_forecast(lat, lon)
    if error:
        return jsonify({"error": error}), 400

    return jsonify({"forecasts": forecasts, "location": {"lat": lat, "lon": lon}})


@app.route("/api/predict", methods=["POST"])
def predict_generation():
    """Predict solar generation based on system configuration and weather."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    # Get and validate location
    lat, lon, error = get_location_from_request(data)
    if error:
        return jsonify({"error": error}), 400

    # Get system configuration
    system_config = get_system_config_from_request(data)

    # Get weather data
    weather = data.get("weather")
    if weather is None:
        weather, error = get_weather_data(lat, lon)
        if error:
            return jsonify({
                "error": f"Weather API error: {error}",
                "hint": "You can provide weather data manually in the request body",
            }), 400

    # Get timestamp
    timestamp_str = data.get("timestamp")
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    # Make prediction
    predicted_kwh, prediction_method, features = make_prediction(
        weather, system_config, lat, lon, timestamp, model
    )

    return jsonify({
        "prediction": {
            "generation_kwh": round(predicted_kwh, 3),
            "method": prediction_method,
            "timestamp": timestamp.isoformat(),
        },
        "system": system_config,
        "weather": weather,
        "location": {"lat": lat, "lon": lon},
        "features": features,
    })


@app.route("/api/predict/daily", methods=["POST"])
def predict_daily_generation():
    """Predict hourly solar generation for a full day."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    # Get and validate location
    lat, lon, error = get_location_from_request(data)
    if error:
        return jsonify({"error": error}), 400

    # Get system configuration
    system_config = get_system_config_from_request(data)

    # Get base weather data (use defaults if API fails)
    weather, error = get_weather_data(lat, lon)
    if error:
        weather = {
            "air_temperature": 25,
            "relative_humidity": 50,
            "wind_speed": 2,
            "wind_direction": 180,
            "dew_point": 20,
            "clouds": 20,
        }

    # Get date
    date_str = data.get("date")
    if date_str:
        try:
            base_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
        except ValueError:
            base_date = datetime.now().date()
    else:
        base_date = datetime.now().date()

    # Generate hourly predictions
    hourly_predictions = []
    total_kwh = 0

    for hour in range(24):
        timestamp = datetime.combine(base_date, datetime.min.time().replace(hour=hour))
        
        # Make prediction
        predicted_kwh, _, _ = make_prediction(
            weather, system_config, lat, lon, timestamp, model
        )
        
        # Calculate solar elevation
        timestamp_series = pd.Series([timestamp])
        solar_elevation = calculate_solar_elevation(
            pd.Series([lat]), pd.Series([lon]), timestamp_series
        )[0]

        hourly_predictions.append({
            "hour": hour,
            "generation_kwh": round(predicted_kwh, 3),
            "solar_elevation": round(solar_elevation, 2),
        })
        total_kwh += predicted_kwh

    return jsonify({
        "date": base_date.isoformat(),
        "hourly": hourly_predictions,
        "total_kwh": round(total_kwh, 3),
        "system": system_config,
        "location": {"lat": lat, "lon": lon},
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

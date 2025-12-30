import os
import sys
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import calculate_solar_elevation

load_dotenv()

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")


def get_weather_data(lat, lon):
    if not OPENWEATHER_API_KEY:
        return None, "OpenWeatherMap API key not configured"

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        weather = {
            "air_temperature": data["main"]["temp"],
            "relative_humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"].get("deg", 0),
            "dew_point": data["main"].get("dew_point", data["main"]["temp"] - 5),
            "description": data["weather"][0]["description"],
            "clouds": data["clouds"]["all"],
        }

        return weather, None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def get_weather_forecast(lat, lon):
    if not OPENWEATHER_API_KEY:
        return None, "OpenWeatherMap API key not configured"

    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        forecasts = []
        for item in data["list"]:
            forecast = {
                "timestamp": item["dt_txt"],
                "air_temperature": item["main"]["temp"],
                "relative_humidity": item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"],
                "wind_direction": item["wind"].get("deg", 0),
                "clouds": item["clouds"]["all"],
                "description": item["weather"][0]["description"],
            }
            forecasts.append(forecast)

        return forecasts, None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def geocode_location(location_name):
    if not OPENWEATHER_API_KEY:
        return None, "OpenWeatherMap API key not configured"

    url = f"https://api.openweathermap.org/geo/1.0/direct?q={location_name}&limit=1&appid={OPENWEATHER_API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            return None, "Location not found"

        return {
            "lat": data[0]["lat"],
            "lon": data[0]["lon"],
            "name": data[0].get("name", location_name),
            "country": data[0].get("country", ""),
        }, None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def reverse_geocode_coords(lat, lon):
    if not OPENWEATHER_API_KEY:
        return None, "OpenWeatherMap API key not configured"

    url = f"https://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHER_API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            return None, "Location not found"

        return {
            "city": data[0].get("name", ""),
            "region": data[0].get("state", ""),
            "country": data[0].get("country", ""),
            "country_code": data[0].get("country", ""),
        }, None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def prepare_features(
    weather, system_config, lat, lon, timestamp=None, dew_point_temp=None
):
    if timestamp is None:
        timestamp = datetime.now()

    hour = timestamp.hour
    month = timestamp.month
    day_of_year = timestamp.timetuple().tm_yday

    # System Specifications
    kwp = system_config.get("kwp", 5.0)
    num_panels = system_config.get("num_panels", 15)
    panel_rating_w = system_config.get("panel_rating_w", 330)
    inverter_kw = system_config.get("inverter_kw", 5.0)

    # Calculate Solar Elevation
    timestamp_series = pd.Series([timestamp])
    solar_elevation = calculate_solar_elevation(
        pd.Series([lat]), pd.Series([lon]), timestamp_series
    )[0]
    solar_elevation = max(0, solar_elevation)

    # Weather Features
    air_temp = weather.get("air_temperature", 25)
    humidity = weather.get("relative_humidity", 50)
    wind_speed = weather.get("wind_speed", 2)
    wind_direction = weather.get("wind_direction", 180)

    dew_point = dew_point_temp or weather.get("dew_point", air_temp - 5)

    # Derived Weather Features
    temp_deviation = abs(air_temp - 25)
    temp_dew_spread = air_temp - dew_point
    humidity_temp = humidity * air_temp / 100
    wind_cooling_effect = wind_speed * air_temp

    # System Efficiency Ratios
    avg_panel_kw = kwp / (num_panels + 0.001)
    inverter_panel_ratio = inverter_kw / (kwp + 0.001)

    # Cyclical Features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Season (Southern Hemisphere)
    if month in [12, 1, 2]:
        season = 0  # Summer
    elif month in [3, 4, 5]:
        season = 1  # Autumn
    elif month in [6, 7, 8]:
        season = 2  # Winter
    else:
        season = 3  # Spring

    features = {
        "AirTemperature": air_temp,
        "RelativeHumidity": humidity,
        "WindSpeed": wind_speed,
        "WindDirection": wind_direction,
        "kWp": kwp,
        "NumberOfPanels": num_panels,
        "TotalInverterKW": inverter_kw,
        "Hour": hour,
        "Month": month,
        "DayOfYear": day_of_year,
        "HourSin": hour_sin,
        "HourCos": hour_cos,
        "MonthSin": month_sin,
        "MonthCos": month_cos,
        "Season": season,
        "SolarElevation": solar_elevation,
        "TempDeviation": temp_deviation,
        "TempDewSpread": temp_dew_spread,
        "HumidityTemp": humidity_temp,
        "WindCoolingEffect": wind_cooling_effect,
        "AvgPanelKW": avg_panel_kw,
        "InverterPanelRatio": inverter_panel_ratio,
    }

    return features


def estimate_generation_simple(weather, system_config, lat, lon, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()

    kwp = system_config.get("kwp", 5.0)

    # Convert Timestamp to Series for Solar Elevation Calculation
    timestamp_series = pd.Series([timestamp])
    solar_elevation = calculate_solar_elevation(
        pd.Series([lat]), pd.Series([lon]), timestamp_series
    )[0]
    solar_elevation = max(0, solar_elevation)

    if solar_elevation <= 0:
        return 0.0

    # Base Generation Estimation Factors
    elevation_factor = np.sin(np.radians(solar_elevation))

    # Temperature Derating
    temp = weather.get("air_temperature", 25)
    temp_factor = 1 - max(0, (temp - 25) * 0.004)

    # Cloud Factor
    clouds = weather.get("clouds", 0)
    cloud_factor = 1 - (clouds / 100) * 0.75

    # Humidity Factor
    humidity = weather.get("relative_humidity", 50)
    humidity_factor = 1 - max(0, (humidity - 60) / 100) * 0.1

    # Calculate Estimated Generation
    estimated_kwh = (
        kwp * elevation_factor * temp_factor * cloud_factor * humidity_factor
    )

    # Apply System Efficiency
    system_efficiency = 0.80
    estimated_kwh *= system_efficiency

    return max(0, estimated_kwh)


def get_location_from_request(data):
    """Extract and validate location from request data."""
    lat = data.get("lat")
    lon = data.get("lon")
    location = data.get("location", "")

    if location and (lat is None or lon is None):
        geo_result, error = geocode_location(location)
        if error:
            return None, None, f"Geocoding error: {error}"
        lat = geo_result["lat"]
        lon = geo_result["lon"]

    if lat is None or lon is None:
        return None, None, "Location (lat/lon or location name) is required"

    return lat, lon, None


def get_system_config_from_request(data):
    """Extract system configuration from request and calculate kWp if needed."""
    system_config = {
        "num_panels": data.get("num_panels", 15),
        "panel_rating_w": data.get("panel_rating_w", 330),
        "inverter_kw": data.get("inverter_kw", 5.0),
        "kwp": data.get("kwp"),
    }

    if system_config["kwp"] is None:
        system_config["kwp"] = (
            system_config["num_panels"] * system_config["panel_rating_w"]
        ) / 1000

    return system_config


def make_prediction(weather, system_config, lat, lon, timestamp, model):
    """Make a single prediction using ML model or fallback to simple estimation."""
    features = prepare_features(weather, system_config, lat, lon, timestamp)

    if model is not None:
        try:
            feature_df = pd.DataFrame([features])
            expected_features = (
                model.feature_names_in_
                if hasattr(model, "feature_names_in_")
                else list(features.keys())
            )
            feature_df = feature_df[expected_features]

            capacity_factor = model.predict(feature_df)[0]
            predicted_kwh = capacity_factor * system_config["kwp"]
            prediction_method = "ml_model"
        except Exception as e:
            predicted_kwh = estimate_generation_simple(
                weather, system_config, lat, lon, timestamp
            )
            prediction_method = "simple_estimation"
            print(f"Model prediction failed, using simple estimation: {e}")
    else:
        predicted_kwh = estimate_generation_simple(
            weather, system_config, lat, lon, timestamp
        )
        prediction_method = "simple_estimation"

    return max(0, predicted_kwh), prediction_method, features

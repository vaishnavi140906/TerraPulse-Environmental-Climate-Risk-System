import os
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import requests
from dotenv import load_dotenv
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.json")

app = Flask(__name__)

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

risk_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "risk_model.keras")
)
risk_scaler = joblib.load(os.path.join(MODEL_DIR, "risk_scaler.pkl"))
risk_encoders = joblib.load(os.path.join(MODEL_DIR, "risk_encoders.pkl"))

climate_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "climate_lstm.keras")
)
climate_scaler = joblib.load(os.path.join(MODEL_DIR, "climate_scaler.pkl"))

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(record):
    history = load_history()
    history.insert(0, record)
    history = history[:50]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def risk_points(label):
    mapping = {
        "Low": 20,
        "Good": 10,
        "Normal": 10,
        "Moderate": 50,
        "Medium": 55,
        "Warning": 65,
        "Poor": 80,
        "High": 90,
        "Heatwave": 95,
    }
    return mapping.get(label, 50)


def overall_score(results):
    values = [
        risk_points(results["air_quality"]),
        risk_points(results["flood_risk"]),
        risk_points(results["fire_risk"]),
        risk_points(results["heatwave_risk"]),
        risk_points(results["drought_risk"]),
    ]
    return round(float(np.mean(values)), 1)


def overall_label(score):
    if score < 35:
        return "Low"
    if score < 65:
        return "Moderate"
    return "High"


def recommendations(results, inputs):
    tips = []

    # 1. Temperature
    temperature = inputs["temperature"]
    if temperature >= 40:
        tips.append({
            "title": "Extreme temperature",
            "status": "High",
            "message": "Avoid strenuous outdoor activities during the hottest hours. Drink water regularly and seek a cool place."
        })
    elif temperature >= 35:
        tips.append({
            "title": "High temperature",
            "status": "Moderate",
            "message": "Stay hydrated, use shade, and reduce prolonged outdoor activity during hot hours."
        })
    else:
        tips.append({
            "title": "Temperature",
            "status": "Normal",
            "message": "Continue monitoring temperature changes and maintain comfortable indoor conditions."
        })

    # 2. Humidity
    humidity = inputs["humidity"]
    if humidity >= 80:
        tips.append({
            "title": "High humidity",
            "status": "High",
            "message": "Improve indoor ventilation where possible and take breaks during physical activity."
        })
    elif humidity <= 30:
        tips.append({
            "title": "Low humidity",
            "status": "Moderate",
            "message": "Drink enough water and watch for dry skin or throat irritation."
        })
    else:
        tips.append({
            "title": "Humidity",
            "status": "Normal",
            "message": "Maintain good ventilation and continue monitoring humidity."
        })

    # 3. Rainfall
    rainfall = inputs["rainfall"]
    if rainfall >= 100:
        tips.append({
            "title": "Very heavy rainfall",
            "status": "High",
            "message": "Check official weather and flood alerts. Avoid crossing flooded roads or flowing water."
        })
    elif rainfall >= 50:
        tips.append({
            "title": "Heavy rainfall",
            "status": "Moderate",
            "message": "Keep drains clear where safe and monitor nearby water levels and weather alerts."
        })
    else:
        tips.append({
            "title": "Rainfall",
            "status": "Normal",
            "message": "Monitor rainfall trends and use water efficiently."
        })

    # 4. Wind speed
    wind_speed = inputs["wind_speed"]
    if wind_speed >= 40:
        tips.append({
            "title": "Strong winds",
            "status": "High",
            "message": "Secure loose outdoor objects and follow official severe-weather instructions."
        })
    elif wind_speed >= 25:
        tips.append({
            "title": "Elevated wind speed",
            "status": "Moderate",
            "message": "Use caution around loose objects, trees, and exposed outdoor areas."
        })
    else:
        tips.append({
            "title": "Wind speed",
            "status": "Normal",
            "message": "Continue monitoring wind conditions, especially when weather changes."
        })

    # 5. Air pollution
    pm25 = inputs["pm25"]
    pm10 = inputs["pm10"]
    if pm25 >= 55 or pm10 >= 150:
        tips.append({
            "title": "Elevated particulate pollution",
            "status": "High",
            "message": "Reduce prolonged outdoor exertion and check local air-quality advisories."
        })
    elif pm25 >= 15 or pm10 >= 45:
        tips.append({
            "title": "Particulate pollution",
            "status": "Moderate",
            "message": "Monitor air quality. Sensitive people should pay attention to symptoms and local guidance."
        })
    else:
        tips.append({
            "title": "Particulate pollution",
            "status": "Normal",
            "message": "Continue monitoring PM2.5 and PM10 levels."
        })

    # 6. Atmospheric pressure
    pressure = inputs["pressure"]
    if pressure < 990:
        tips.append({
            "title": "Low atmospheric pressure",
            "status": "Monitor",
            "message": "Low pressure alone does not confirm a storm. Check official weather forecasts for developing conditions."
        })
    else:
        tips.append({
            "title": "Atmospheric pressure",
            "status": "Monitor",
            "message": "Track pressure changes alongside official weather forecasts."
        })

    # 7. Water level
    water_level = inputs["water_level"]
    if water_level >= 5:
        tips.append({
            "title": "Elevated water level",
            "status": "High",
            "message": "If this measurement represents a local river or reservoir, compare it with official warning levels and follow local instructions."
        })
    elif water_level >= 3:
        tips.append({
            "title": "Water level",
            "status": "Monitor",
            "message": "Monitor changes against the relevant local gauge and official flood guidance."
        })
    else:
        tips.append({
            "title": "Water level",
            "status": "Normal",
            "message": "Continue monitoring water levels, especially during rainfall."
        })

    # 8. Soil moisture
    soil_moisture = inputs["soil_moisture"]
    if soil_moisture < 20:
        tips.append({
            "title": "Low soil moisture",
            "status": "High",
            "message": "Check plants and soil before watering. Water efficiently near the root zone when needed."
        })
    elif soil_moisture < 35:
        tips.append({
            "title": "Soil moisture declining",
            "status": "Moderate",
            "message": "Monitor soil conditions and consider efficient irrigation if crops or plants need water."
        })
    else:
        tips.append({
            "title": "Soil moisture",
            "status": "Normal",
            "message": "Continue monitoring moisture and avoid unnecessary watering."
        })

    # 9. Model-predicted hazards
    if results["flood_risk"] in ["High", "Warning"]:
        tips.append({
            "title": "Flood-risk prediction",
            "status": "High",
            "message": "Check official flood warnings and avoid floodwater. The model output is not an official alert."
        })

    if results["fire_risk"] in ["High", "Warning"]:
        tips.append({
            "title": "Forest-fire prediction",
            "status": "High",
            "message": "Avoid open flames near dry vegetation and check official fire-risk information."
        })

    if results["heatwave_risk"] in ["Heatwave", "Warning", "High"]:
        tips.append({
        "title": "Heatwave prediction",
        "status": "High",
        "message": "The model indicates elevated heatwave risk. Stay hydrated, seek shade, and check official heat-health advisories."
    })
    elif results["heatwave_risk"] in ["Moderate", "Medium"]:
        tips.append({
        "title": "Heatwave prediction",
        "status": "Moderate",
        "message": "Monitor temperature conditions and follow local weather updates."
    })
    else:
        tips.append({
        "title": "Heatwave prediction",
        "status": "Low",
        "message": "The model does not indicate an elevated heatwave category. Continue monitoring weather conditions."
    })

    if results["drought_risk"] in ["High", "Warning"]:
        tips.append({
            "title": "Drought-risk prediction",
            "status": "High",
            "message": "Use water efficiently and check local water-supply and drought advisories."
        })

    return tips

def get_float(name, default=0.0):
    try:
        return float(request.form.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    temperature = get_float("temperature", 29)
    humidity = get_float("humidity", 65)
    rainfall = get_float("rainfall", 20)
    wind_speed = get_float("wind_speed", 15)
    pressure = get_float("pressure", 1012)
    pm25 = get_float("pm25", 40)
    pm10 = get_float("pm10", 65)
    water_level = get_float("water_level", 3)
    soil_moisture = get_float("soil_moisture", 45)

    features = np.array([[
        temperature,
        humidity,
        rainfall,
        wind_speed,
        pressure,
        pm25,
        pm10,
        water_level,
        soil_moisture
    ]], dtype="float32")

    scaled = risk_scaler.transform(features)

    predictions = risk_model.predict(scaled, verbose=0)

    flood_idx = int(np.argmax(predictions[0][0]))
    fire_idx = int(np.argmax(predictions[1][0]))
    heat_idx = int(np.argmax(predictions[2][0]))
    drought_idx = int(np.argmax(predictions[3][0]))

    flood_label = risk_encoders["flood"].inverse_transform([flood_idx])[0]
    fire_label = risk_encoders["fire"].inverse_transform([fire_idx])[0]
    heat_label = risk_encoders["heatwave"].inverse_transform([heat_idx])[0]
    drought_label = risk_encoders["drought"].inverse_transform([drought_idx])[0]
    
    print("Heatwave classes:", risk_encoders["heatwave"].classes_)
    print("Heatwave output:", predictions[2][0])
    print("Predicted heatwave label:", heat_label)

    # Air quality is calculated from the environmental input.
    air_score = (
        0.75 * pm25
        + 0.25 * pm10
        + 0.15 * temperature
        - 0.08 * wind_speed
    )

    if air_score < 45:
        air_label = "Good"
    elif air_score < 90:
        air_label = "Moderate"
    else:
        air_label = "Poor"

    # LSTM forecast uses the saved sequence and adjusts the final
    # timestep using the user's current values.
    current_row = np.array([[
        temperature, humidity, rainfall, wind_speed, pressure
    ]], dtype="float32")

    current_scaled = climate_scaler.transform(current_row)[0]

    sequence = np.load(
        os.path.join(MODEL_DIR, "last_sequence.npy")
    ).copy()

    sequence = np.vstack([sequence[1:], current_scaled])

    forecast_scaled = climate_model.predict(
        sequence.reshape(1, sequence.shape[0], sequence.shape[1]),
        verbose=0
    )[0]

    # Inverse only temperature and rainfall by building a 5-column row.
    forecast_full = np.zeros((1, 5), dtype="float32")
    forecast_full[0, 0] = forecast_scaled[0]
    forecast_full[0, 2] = forecast_scaled[1]

    forecast_full_original = climate_scaler.inverse_transform(forecast_full)[0]

    predicted_temperature = round(float(forecast_full_original[0]), 2)
    predicted_rainfall = round(max(0.0, float(forecast_full_original[2])), 2)

    results = {
        "air_quality": air_label,
        "flood_risk": flood_label,
        "fire_risk": fire_label,
        "heatwave_risk": heat_label,
        "drought_risk": drought_label,
    }

    score = overall_score(results)
    overall = overall_label(score)

    results["overall_score"] = score
    results["overall_risk"] = overall
    results["predicted_temperature"] = predicted_temperature
    results["predicted_rainfall"] = predicted_rainfall
 

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "wind_speed": wind_speed,
            "pressure": pressure,
            "pm25": pm25,
            "pm10": pm10,
            "water_level": water_level,
            "soil_moisture": soil_moisture
        },
        "results": results
    }

    results["recommendations"] = recommendations(results, record["inputs"])
    record["results"] = results

    save_history(record)

    return render_template(
        "index.html",
        results=results,
        input_values=record["inputs"],
        history=load_history()
    )


@app.route("/history")
def history():
    return jsonify(load_history())

@app.route("/api/weather")
def get_live_weather():
    city = request.args.get("city", "Chennai").strip()
    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        return jsonify({"error": "Weather API key is not configured"}), 500

    if not city:
        return jsonify({"error": "Please enter a city name"}), 400

    try:
        response = requests.get(
            "https://api.weatherapi.com/v1/current.json",
            params={
                "key": api_key,
                "q": city,
                "aqi": "yes"
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        current = data["current"]
        location = data["location"]
        air = current.get("air_quality") or {}

        return jsonify({
            "city": location["name"],
            "country": location["country"],
            "temperature": current["temp_c"],
            "humidity": current["humidity"],
            "rainfall": current["precip_mm"],
            "wind_speed": current["wind_kph"],
            "pressure": current["pressure_mb"],
            "pm25": air.get("pm2_5"),
            "pm10": air.get("pm10"),
            "condition": current["condition"]["text"],
            "updated_at": current["last_updated"],
            "source": "WeatherAPI.com"
        })

    except requests.HTTPError as e:
        print("Weather API HTTP error:", response.status_code, response.text)
        return jsonify({
            "error": f"Weather API error: {response.status_code}. Check the API key, city, or API limits."
        }), 502

    except requests.RequestException as e:
        print("Weather connection error:", str(e))
        return jsonify({
            "error": "Could not connect to the weather service."
        }), 502

    except (KeyError, ValueError) as e:
        print("Weather response parsing error:", str(e))
        return jsonify({
            "error": "Unexpected response from the weather service."
        }), 502


if __name__ == "__main__":
    if not os.path.exists(os.path.join(MODEL_DIR, "risk_model.keras")):
        raise FileNotFoundError(
            "Models are missing. Run 'python train_models.py' first."
        )

    app.run(debug=True)

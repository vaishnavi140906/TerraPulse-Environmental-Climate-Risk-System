import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. SYNTHETIC CLIMATE DATA
# ---------------------------------------------------------
# Features:
# temperature, humidity, rainfall, wind_speed, pressure,
# pm25, pm10, water_level, soil_moisture
# ---------------------------------------------------------

N = 7000

temperature = np.random.normal(29, 6, N).clip(12, 48)
humidity = np.random.normal(65, 18, N).clip(15, 100)
rainfall = np.random.gamma(2.0, 18, N).clip(0, 250)
wind_speed = np.random.normal(18, 9, N).clip(0, 70)
pressure = np.random.normal(1012, 10, N).clip(980, 1040)
pm25 = np.random.normal(48, 35, N).clip(5, 250)
pm10 = (pm25 * 1.45 + np.random.normal(0, 18, N)).clip(10, 400)
water_level = np.random.normal(3.2, 1.5, N).clip(0, 12)
soil_moisture = np.random.normal(45, 20, N).clip(3, 95)

df = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
    "wind_speed": wind_speed,
    "pressure": pressure,
    "pm25": pm25,
    "pm10": pm10,
    "water_level": water_level,
    "soil_moisture": soil_moisture,
})

# ---------------------------------------------------------
# 2. TARGETS
# These formulas create a realistic demonstration dataset.
# ---------------------------------------------------------

# Air quality score: higher means worse air quality
air_score = (
    0.75 * pm25
    + 0.25 * pm10
    + 0.15 * temperature
    - 0.08 * wind_speed
)
air_score = np.clip(air_score, 0, None)

air_quality = np.where(
    air_score < 45, "Good",
    np.where(air_score < 90, "Moderate", "Poor")
)

# Flood score
flood_score = (
    0.045 * rainfall
    + 0.65 * water_level
    + 0.015 * humidity
    + 0.008 * rainfall * water_level
)
flood_risk = np.where(
    flood_score < 4.5, "Low",
    np.where(flood_score < 8.0, "Medium", "High")
)

# Fire score
fire_score = (
    0.10 * temperature
    + 0.06 * wind_speed
    - 0.035 * humidity
    - 0.025 * rainfall
    + 0.015 * (100 - soil_moisture)
)
fire_risk = np.where(
    fire_score < 2.5, "Low",
    np.where(fire_score < 5.0, "Medium", "High")
)

# Heatwave score
# Heatwave score
# Simplified rules for this demonstration project
heat_score = temperature + 0.03 * np.maximum(humidity - 50, 0)

heat_risk = np.where(
    heat_score >= 40, "Heatwave",
    np.where(heat_score >= 35, "Warning", "Normal")
)

# Drought score
drought_score = (
    0.06 * temperature
    - 0.025 * rainfall
    - 0.018 * humidity
    - 0.025 * soil_moisture
)
drought_risk = np.where(
    drought_score < 0.9, "Low",
    np.where(drought_score < 1.9, "Medium", "High")
)

# Add small noise to prevent a perfectly deterministic mapping
df["air_target"] = air_score + np.random.normal(0, 4, N)
df["rainfall_target"] = rainfall * 1.04 + np.random.normal(0, 5, N)
df["temperature_target"] = temperature + np.random.normal(0, 0.7, N)

risk_df = pd.DataFrame({
    "flood": flood_risk,
    "fire": fire_risk,
    "heatwave": heat_risk,
    "drought": drought_risk,
})

risk_features = df[
    ["temperature", "humidity", "rainfall", "wind_speed",
     "pressure", "pm25", "pm10", "water_level", "soil_moisture"]
].values

risk_encoder = LabelEncoder()
# Encode the four target groups separately because each has its own labels.
flood_encoder = LabelEncoder()
fire_encoder = LabelEncoder()
heat_encoder = LabelEncoder()
drought_encoder = LabelEncoder()

y_flood = flood_encoder.fit_transform(flood_risk)
y_fire = fire_encoder.fit_transform(fire_risk)
y_heat = heat_encoder.fit_transform(heat_risk)
y_drought = drought_encoder.fit_transform(drought_risk)

# ---------------------------------------------------------
# 3. RISK MODEL
# Multi-output neural network
# ---------------------------------------------------------

risk_scaler = StandardScaler()
X_risk = risk_scaler.fit_transform(risk_features).astype("float32")

risk_input = Input(shape=(X_risk.shape[1],), name="environment_input")
x = Dense(128, activation="relu")(risk_input)
x = Dropout(0.25)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.15)(x)
x = Dense(32, activation="relu")(x)

flood_output = Dense(len(flood_encoder.classes_), activation="softmax", name="flood")(x)
fire_output = Dense(len(fire_encoder.classes_), activation="softmax", name="fire")(x)
heat_output = Dense(len(heat_encoder.classes_), activation="softmax", name="heatwave")(x)
drought_output = Dense(len(drought_encoder.classes_), activation="softmax", name="drought")(x)

risk_model = tf.keras.Model(
    inputs=risk_input,
    outputs=[flood_output, fire_output, heat_output, drought_output]
)

risk_model.compile(
    optimizer="adam",
    loss={
        "flood": "sparse_categorical_crossentropy",
        "fire": "sparse_categorical_crossentropy",
        "heatwave": "sparse_categorical_crossentropy",
        "drought": "sparse_categorical_crossentropy"
    },
    metrics={
        "flood": ["accuracy"],
        "fire": ["accuracy"],
        "heatwave": ["accuracy"],
        "drought": ["accuracy"]
    }
)

risk_model.fit(
    X_risk,
    {
        "flood": y_flood,
        "fire": y_fire,
        "heatwave": y_heat,
        "drought": y_drought,
    },
    epochs=18,
    batch_size=64,
    validation_split=0.15,
    callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
    verbose=1,
)

risk_model.save(os.path.join(MODEL_DIR, "risk_model.keras"))
joblib.dump(risk_scaler, os.path.join(MODEL_DIR, "risk_scaler.pkl"))

joblib.dump(
    {
        "flood": flood_encoder,
        "fire": fire_encoder,
        "heatwave": heat_encoder,
        "drought": drought_encoder,
    },
    os.path.join(MODEL_DIR, "risk_encoders.pkl")
)

# ---------------------------------------------------------
# 4. LSTM CLIMATE FORECAST MODEL
# Predicts next rainfall and temperature.
# ---------------------------------------------------------

SEQ_LEN = 12

climate_data = df[
    ["temperature", "humidity", "rainfall", "wind_speed", "pressure"]
].values.astype("float32")

climate_scaler = StandardScaler()
scaled_climate = climate_scaler.fit_transform(climate_data)

X_seq = []
y_seq = []

for i in range(SEQ_LEN, len(scaled_climate)):
    X_seq.append(scaled_climate[i - SEQ_LEN:i])
    # Target: next temperature and rainfall
    y_seq.append(scaled_climate[i, [0, 2]])

X_seq = np.array(X_seq, dtype="float32")
y_seq = np.array(y_seq, dtype="float32")

lstm_model = Sequential([
    Input(shape=(SEQ_LEN, 5)),
    LSTM(64, return_sequences=True),
    Dropout(0.20),
    LSTM(32),
    Dense(32, activation="relu"),
    Dense(2)
])

lstm_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

lstm_model.fit(
    X_seq,
    y_seq,
    epochs=15,
    batch_size=64,
    validation_split=0.15,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    verbose=1,
)

lstm_model.save(os.path.join(MODEL_DIR, "climate_lstm.keras"))
joblib.dump(climate_scaler, os.path.join(MODEL_DIR, "climate_scaler.pkl"))

# Save a final sample sequence so the application can make an LSTM forecast
np.save(os.path.join(MODEL_DIR, "last_sequence.npy"), X_seq[-1])

print("\n==============================================")
print("MODEL TRAINING COMPLETED")
print("==============================================")
print("Created:")
print("- models/risk_model.keras")
print("- models/risk_scaler.pkl")
print("- models/risk_encoders.pkl")
print("- models/climate_lstm.keras")
print("- models/climate_scaler.pkl")
print("- models/last_sequence.npy")
print("==============================================")

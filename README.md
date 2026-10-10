# Deep Learning-Based Environmental and Climate Risk Prediction System

A Flask + TensorFlow/Keras dashboard that predicts environmental and climate risks from weather and environmental factors.

## Modules
- Air Quality Prediction
- Rainfall Prediction
- Temperature Prediction
- Flood Risk
- Forest Fire Risk
- Heatwave Risk
- Drought Risk
- Overall Environmental Risk Score
- AI Recommendations
- Prediction History

## Project structure

```text
Environmental_Climate_Risk_System/
│
├── app.py
├── train_models.py
├── requirements.txt
├── prediction_history.json
│
├── models/
│   ├── climate_lstm.keras
│   ├── risk_model.keras
│   ├── climate_scaler.pkl
│   ├── risk_scaler.pkl
│   └── risk_label_encoder.pkl
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

## Run in VS Code

Open the project folder in VS Code terminal:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_models.py
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The training script creates the model files inside `models/`.

## Important
The included training data is synthetic climate-style data so the project can run immediately without downloading a dataset. For a final academic version, replace the synthetic generator with a real historical climate/environment dataset and retrain the models.

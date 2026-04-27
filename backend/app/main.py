from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_pipeline", "artifacts", "disruption_model.joblib")
model = joblib.load(MODEL_PATH)

class PredictionInput(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict_disruption(data: PredictionInput):
    input_data = np.array([data.features])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {
        "disruption": bool(prediction),
        "probability": f"{round(probability * 100, 2)}%"
    }

@app.get("/")
def health_check():
    return {"status": "Backend is running"}
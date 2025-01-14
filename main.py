from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Sales Prediction API", description="API for predicting Rossmann store sales")

# Load the serialized model
model_path = "data\model-2025-01-14-21-33-22.pkl"
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Model file not found. Ensure the serialized model exists in the same directory.")

# Define input schema
class PredictionInput(BaseModel):
    store_id: int
    promo: int
    state_holiday: str
    school_holiday: int
    competition_distance: float
    year: int
    month: int
    day: int
    week_of_year: int
    day_of_week: int

# Define the /predict endpoint
@app.post("/predict", summary="Predict Sales")
async def predict_sales(input_data: PredictionInput):
    try:
        # Convert input data into a feature array
        features = np.array([
            input_data.store_id,
            input_data.promo,
            input_data.state_holiday,
            input_data.school_holiday,
            input_data.competition_distance,
            input_data.year,
            input_data.month,
            input_data.day,
            input_data.week_of_year,
            input_data.day_of_week
        ]).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)

        # Return the prediction as JSON
        return {"predicted_sales": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

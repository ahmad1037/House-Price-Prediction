from fastapi import FastAPI
from app.schemas import HouseFeatures
from app.predictor import predict
app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
)
@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is running."
    }

@app.post("/predict")
def predict_house_price(
    features: HouseFeatures,
):
    """
    Predict house price.
    """

    prediction = predict(
        features.model_dump()
    )

    return {
        "predicted_price": prediction
    }

from app.schemas import (
    HouseFeatures,
    PredictionResponse,
)

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_house_price(
    features: HouseFeatures,
):
    prediction = predict(
        features.model_dump()
    )

    return PredictionResponse(
        predicted_price=prediction
    )
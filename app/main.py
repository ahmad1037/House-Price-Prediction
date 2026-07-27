from fastapi import FastAPI

from app.schemas import (
    HouseFeatures,
    PredictionResponse,
)

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


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_house_price(
    features: HouseFeatures,
):
    """
    Predict house price from house features.
    """

    prediction = predict(
        {
            "Overall Qual": features.OverallQual,
            "Gr Liv Area": features.GrLivArea,
            "Garage Cars": features.GarageCars,
            "Total Bsmt SF": features.TotalBsmtSF,
            "Year Built": features.YearBuilt,
        }
    )

    return PredictionResponse(
        predicted_price=prediction
    )
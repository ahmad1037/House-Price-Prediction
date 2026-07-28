from app.predictor import predict

def test_prediction_returns_float():

    sample = {
        "Overall Qual": 7,
        "Gr Liv Area": 1710,
        "Garage Cars": 2,
        "Total Bsmt SF": 856,
        "Year Built": 2003,
    }

    result = predict(sample)

    assert isinstance(
        result,
        float,
    )
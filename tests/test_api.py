from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    assert "message" in response.json()

def test_prediction_endpoint():

    payload = {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "GarageCars": 2,
        "TotalBsmtSF": 856,
        "YearBuilt": 2003,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_price" in data

    assert isinstance(
        data["predicted_price"],
        float,
    )

def test_invalid_input():

    payload = {
        "OverallQual": -5,
        "GrLivArea": -100,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Bike Rental Prediction API is running."


def test_prediction():

    payload = {
        "season": 2,
        "yr": 1,
        "mnth": 6,
        "hr": 18,
        "holiday": 0,
        "weekday": 2,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.62,
        "atemp": 0.60,
        "hum": 0.55,
        "windspeed": 0.19
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "model_version" in data
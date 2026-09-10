from fastapi import FastAPI
import joblib
import pandas as pd

from api.schemas import BikeRentalRequest

app = FastAPI(
    title="Bike Rental Prediction API",
    version="1.0"
)

# Load trained model
model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {
        "message": "Bike Rental Prediction API is running."
    }


@app.post("/predict")
def predict(data: BikeRentalRequest):

    df = pd.DataFrame([data.model_dump()])

    # Same feature engineering used during training
    df["rush_hour"] = df["hr"].apply(
        lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0
    )

    df["is_weekend"] = df["weekday"].apply(
        lambda x: 1 if x in [0, 6] else 0
    )

    df["temp_diff"] = df["atemp"] - df["temp"]

    df["bad_weather"] = df["weathersit"].apply(
        lambda x: 1 if x >= 3 else 0
    )

    df["hum_wind"] = df["hum"] * df["windspeed"]

    prediction = model.predict(df)[0]

    return {
        "prediction": round(float(prediction), 2),
        "model_version": "1.0"
    }
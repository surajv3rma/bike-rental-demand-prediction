"""
Feature engineering functions for the Bike Rental Demand Prediction project.
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for model training and inference.
    """

    df = df.copy()

    # Rush hour indicator
    df["rush_hour"] = df["hr"].apply(
        lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0
    )

    # Weekend indicator
    df["is_weekend"] = df["weekday"].apply(
        lambda x: 1 if x in [0, 6] else 0
    )

    # Difference between actual and feels-like temperature
    df["temp_diff"] = df["atemp"] - df["temp"]

    # Poor weather indicator
    df["bad_weather"] = df["weathersit"].apply(
        lambda x: 1 if x >= 3 else 0
    )

    # Interaction feature
    df["hum_wind"] = df["hum"] * df["windspeed"]

    return df
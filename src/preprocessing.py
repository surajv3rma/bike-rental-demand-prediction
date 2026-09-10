import pandas as pd
from pathlib import Path


def load_data(file_path):
    """
    Load Bike Sharing dataset.
    """
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Remove duplicate and missing records.
    """
    df = df.drop_duplicates()
    df = df.dropna()

    return df


def engineer_features(df):
    """
    Shared feature engineering used by BOTH
    training and inference.
    """

    df = df.copy()

    # Rush hour
    df["rush_hour"] = df["hr"].apply(
        lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0
    )

    # Weekend
    df["is_weekend"] = df["weekday"].apply(
        lambda x: 1 if x in [0, 6] else 0
    )

    # Temperature difference
    df["temp_diff"] = df["atemp"] - df["temp"]

    # Weather severity
    df["bad_weather"] = df["weathersit"].apply(
        lambda x: 1 if x >= 3 else 0
    )

    # Humidity × Wind
    df["hum_wind"] = (
        df["hum"] *
        df["windspeed"]
    )

    return df


def save_processed_data(df, output_path):
    """
    Save processed dataset.
    """
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    df = load_data("data/raw/hour.csv")

    df = clean_data(df)

    df = engineer_features(df)

    save_processed_data(
        df,
        "data/processed/hour_processed.csv"
    )

    print("Preprocessing completed successfully.")
import pandas as pd

TRAIN_DATA = "data/raw/hour.csv"
RECENT_DATA = "data/raw/hour.csv"   # We'll change this later if needed

DRIFT_THRESHOLD = 0.15


def check_missing_values(df):
    """
    Check for missing values.
    """
    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("✅ No missing values found.")
    else:
        print("⚠ Missing Values:")
        print(missing[missing > 0])


def check_feature_drift(train_df, recent_df, feature):
    """
    Compare mean values between training data and recent data.
    """

    train_mean = train_df[feature].mean()
    recent_mean = recent_df[feature].mean()

    drift = abs(recent_mean - train_mean) / train_mean

    print(f"\nFeature: {feature}")
    print(f"Training Mean : {train_mean:.4f}")
    print(f"Recent Mean   : {recent_mean:.4f}")
    print(f"Drift Score   : {drift:.4f}")

    if drift > DRIFT_THRESHOLD:
        print("⚠ WARNING: Feature drift detected!")
    else:
        print("✅ No significant drift detected.")


def main():

    train_df = pd.read_csv(TRAIN_DATA)
    recent_df = pd.read_csv(RECENT_DATA)

    print("========== DATA QUALITY ==========")

    check_missing_values(train_df)

    print("\n========== FEATURE DRIFT ==========")

    for feature in [
        "temp",
        "hum",
        "windspeed"
    ]:
        check_feature_drift(
            train_df,
            recent_df,
            feature
        )


if __name__ == "__main__":
    main()
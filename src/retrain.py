"""
Retraining trigger logic for the Bike Rental Demand Prediction project.
This script simulates when a production system should retrain the model.
"""

NEW_DATA_DAYS = 35
MODEL_RMSE = 52.0
PREVIOUS_RMSE = 41.9
DRIFT_SCORE = 0.18

MAX_NEW_DATA_DAYS = 30
RMSE_DEGRADATION_THRESHOLD = 10.0
DRIFT_THRESHOLD = 0.15


def should_retrain():
    """
    Decide whether retraining should be triggered.
    """

    reasons = []

    # Rule 1: Enough new data collected
    if NEW_DATA_DAYS >= MAX_NEW_DATA_DAYS:
        reasons.append(
            f"New data available for {NEW_DATA_DAYS} days."
        )

    # Rule 2: Model performance degraded
    if (MODEL_RMSE - PREVIOUS_RMSE) >= RMSE_DEGRADATION_THRESHOLD:
        reasons.append(
            "Model RMSE degraded significantly."
        )

    # Rule 3: Feature drift detected
    if DRIFT_SCORE >= DRIFT_THRESHOLD:
        reasons.append(
            "Feature drift exceeded threshold."
        )

    if reasons:
        print("Retraining Required")
        print("-" * 40)

        for reason in reasons:
            print(f"- {reason}")

    else:
        print("No retraining required.")


if __name__ == "__main__":
    should_retrain()
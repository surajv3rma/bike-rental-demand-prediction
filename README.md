# Bike Rental Demand Prediction – Mini Production ML System

## Project Overview

This project implements a mini production Machine Learning system for predicting hourly bike rental demand using the UCI Bike Sharing Dataset. It demonstrates the complete lifecycle of an ML application, including data preprocessing, feature engineering, model training, model serving through a REST API, batch data ingestion, monitoring, drift detection, retraining logic, latency measurement, and basic testing.

---

## Project Structure

```
bike-demand-production-ml/

├── api/
├── artifacts/
├── config/
├── data/
├── ingestion/
├── models/
├── src/
├── tests/
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python 3.10
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Pytest
* Joblib

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preprocessing

Run:

```bash
python src/preprocessing.py
```

This script:

* Loads the raw dataset
* Cleans the data
* Performs feature engineering
* Saves the processed dataset

---

## Train the Model

Run:

```bash
python src/train.py
```

The training pipeline:

* Loads processed data
* Trains a baseline Linear Regression model
* Trains a Random Forest model
* Compares evaluation metrics
* Saves the best model
* Generates an evaluation report

---

## Run the Prediction API

Start the FastAPI server:

```bash
python -m uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

to access the interactive Swagger documentation.

---

## Batch Data Ingestion

Run:

```bash
python ingestion/ingest.py
```

The ingestion pipeline:

* Reads new CSV files
* Merges them with the master dataset
* Logs ingestion details
* Archives processed files

---

## Drift Detection

Run:

```bash
python src/drift.py
```

This script checks:

* Missing values
* Basic feature drift
* Data quality issues

---

## Retraining Logic

Run:

```bash
python src/retrain.py
```

The retraining policy is triggered when:

* Sufficient new data is available
* Model performance degrades
* Feature drift exceeds the configured threshold

---

## API Performance Test

Run:

```bash
python tests/latency_test.py
```

This reports:

* Average latency
* P95 latency

---

## Unit Tests

Run:

```bash
python -m pytest
```

---

## Model

**Baseline:** Linear Regression

**Candidate:** Random Forest Regressor

**Selected Model:** Random Forest

Evaluation metrics include:

* RMSE
* MAE
* R² Score

---

## Future Improvements

* Shared feature store for training and inference
* Automated scheduled retraining
* Model registry integration
* Advanced drift detection
* Cloud deployment using Docker and Kubernetes

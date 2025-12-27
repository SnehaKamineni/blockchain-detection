import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os

def preprocess_data(df):
    """
    Keep only numeric columns and fill missing values.
    """
    numeric_df = df.select_dtypes(include=np.number).fillna(0)
    return numeric_df

def detect_anomalies(X, model_path="model.pkl"):
    """
    Detect anomalies using Isolation Forest.
    - If model.pkl exists, use it.
    - Otherwise, train a new model.
    Returns a DataFrame with 'is_anomaly' column (True/False)
    """
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        # Save model for later
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    preds = model.predict(X)
    results = X.copy()
    results['anomaly'] = preds
    results['is_anomaly'] = results['anomaly'] == -1
    return results

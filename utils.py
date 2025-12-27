import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def preprocess_data(df):
    """
    Keep only numeric columns and fill missing values
    """
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.fillna(0)
    return numeric_df


def detect_anomalies(X):
    """
    Detect anomalies using Isolation Forest
    Returns dataframe with anomaly labels
    """
    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    preds = model.fit_predict(X)

    result = X.copy()
    result["anomaly"] = preds
    result["is_anomaly"] = result["anomaly"] == -1

    return result

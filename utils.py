import pandas as pd
import numpy as np

def preprocess_data(data):
    """
    Example preprocessing:
    - Fill missing numeric values with 0
    - Keep numeric columns only
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df[numeric_cols]
    else:
        return pd.read_csv(data)

def detect_anomalies(data, model=None):
    """
    Detect anomalies using either ML model (if provided) or simple rule-based:
    - Rule: any negative value is an anomaly
    Returns: list of row indices that are anomalous
    """
    anomalies = []
    if isinstance(data, pd.DataFrame):
        if model:
            # Example placeholder: model prediction
            preds = model.predict(data)
            anomalies = list(np.where(preds == 1)[0])
        else:
            for idx, row in data.iterrows():
                if any(row < 0):
                    anomalies.append(idx)
    return anomalies

# utils.py

import pandas as pd

def preprocess_data(data):
    """
    Placeholder function to preprocess blockchain data.
    Currently, it just returns the input as a DataFrame.
    Replace this with real preprocessing logic.
    """
    if isinstance(data, pd.DataFrame):
        return data
    else:
        # assume data is a CSV path
        return pd.read_csv(data)

def detect_anomalies(data):
    """
    Placeholder function to detect anomalies in blockchain data.
    Currently, it returns an empty list.
    Replace this with your real anomaly detection logic.
    """
    # Example: mark rows with negative values as anomalies
    anomalies = []
    if isinstance(data, pd.DataFrame):
        for idx, row in data.iterrows():
            if any(row < 0):
                anomalies.append(idx)
    return anomalies

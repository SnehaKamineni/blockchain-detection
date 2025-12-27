import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def preprocess_data(df):
    """
    - Keep numeric columns
    - Fill missing values
    """
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.fillna(0)
    return numeric_df


def detect_anomalies(df):
    """
    Uses Isolation Forest to detect anomalies
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,   # ONLY 5% anomalies
        random_state=42
    )

    preds = model.fit_predict(df)

    result_df = df.copy()
    result_df["is_anomaly"] = preds == -1

    return result_df

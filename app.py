import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import preprocess_data, detect_anomalies

# Page configuration
st.set_page_config(
    page_title="Blockchain Transaction Anomaly Detection",
    layout="wide"
)

st.title("Blockchain Transaction Anomaly Detection Tool 🚀")
st.write("Upload a blockchain transaction CSV file to detect anomalies.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Transaction Data")
    st.dataframe(df.head())

    # Preprocess
    X = preprocess_data(df)

    if X.empty:
        st.error("No numeric columns found in CSV.")
    else:
        # Detect anomalies
        results = detect_anomalies(X)

        st.subheader("Anomaly Detection Results")
        st.dataframe(results)

        # Visualization
        numeric_col = X.columns[0]  # first numeric column

        fig, ax = plt.subplots(figsize=(10, 5))

        # Normal points
        normal = results[results["is_anomaly"] == False]
        ax.scatter(
            normal.index,
            normal[numeric_col],
            color="green",
            label="Normal"
        )

        # Anomalous points
        anomalies = results[results["is_anomaly"] == True]
        ax.scatter(
            anomalies.index,
            anomalies[numeric_col],
            color="red",
            label="Anomaly"
        )

        ax.set_title("Transaction Anomaly Detection")
        ax.set_xlabel("Transaction Index")
        ax.set_ylabel(numeric_col)
        ax.legend()

        st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess_data, detect_anomalies

# Page configuration
st.set_page_config(page_title="Blockchain Detection App", layout="wide")

st.title("Blockchain Detection App 🚀")
st.write("Upload a blockchain transaction CSV file to detect anomalies.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Transaction Data (Preview)")
    st.dataframe(df.head())

    # Preprocess
    processed_df = preprocess_data(df)

    # Detect anomalies
    results = detect_anomalies(processed_df)

    st.subheader("Detection Results")
    st.dataframe(results.head())

    # Visualization
    numeric_col = processed_df.columns[0]

    normal = results[results["is_anomaly"] == False]
    anomalies = results[results["is_anomaly"] == True]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(normal.index, normal[numeric_col], label="Normal", alpha=0.6)
    ax.scatter(anomalies.index, anomalies[numeric_col], label="Anomaly", alpha=0.9)
    ax.set_xlabel("Transaction Index")
    ax.set_ylabel(numeric_col)
    ax.set_title("Blockchain Transaction Anomaly Detection")
    ax.legend()

    st.pyplot(fig)

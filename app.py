import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess_data, detect_anomalies

# Page configuration
st.set_page_config(page_title="Blockchain Detection App", layout="wide")

st.title("Blockchain Detection App 🚀")
st.write("Upload a blockchain transaction CSV file to detect anomalies.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")
    
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Transaction Data")
    st.dataframe(df.head())
    
    # Preprocess data
    X = preprocess_data(df)
    
    # Detect anomalies
    results = detect_anomalies(X)
    
    st.subheader("Detection Results")
    st.dataframe(results)
    
    # Visualize anomalies
    if not X.empty:
        numeric_col = X.columns[0]  # use first numeric column for plotting

        normal = results[results['is_anomaly'] == False]
        anomalies = results[results['is_anomaly'] == True]

        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(normal.index, normal[numeric_col], color='green', label='Normal')
        ax.scatter(anomalies.index, anomalies[numeric_col], color='red', label='Anomaly')
        ax.set_xlabel("Index")
        ax.set_ylabel(numeric_col)
        ax.set_title("Transaction Anomaly Detection")
        ax.legend()
        st.pyplot(fig)

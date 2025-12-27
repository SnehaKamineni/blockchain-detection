import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
    
    # Load ML model (optional)
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.warning("No ML model found. Using rule-based detection.")
        model = None
    
    # Detect anomalies
    results = detect_anomalies(X, model)
    
    st.subheader("Detection Results")
    if isinstance(results, pd.DataFrame):
        st.dataframe(results)
    else:
        st.write(f"Anomalous rows: {results}")
    
    # Visualize anomalies
    if not df.empty:
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]  # pick first numeric column for plotting
            fig, ax = plt.subplots()
            ax.plot(df.index, df[col], label="Transactions")
            # Mark anomalies
            if results:
                ax.scatter(results, df.iloc[results][col], color='red', label="Anomalies")
            ax.set_xlabel("Index")
            ax.set_ylabel(col)
            ax.set_title("Transaction Data with Anomalies")
            ax.legend()
            st.pyplot(fig)

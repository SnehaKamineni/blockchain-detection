import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

    # Load ML model (if using)
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
    st.dataframe(results)

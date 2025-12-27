# anomalies.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    
    # Create Datetime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Drop original Date/Time columns
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    # Replace -200 with NaN and fill missing values
    df.replace(-200, np.nan, inplace=True)
    df.interpolate(method='time', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df

def add_time_features(df):
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df.drop(['hour'], axis=1, inplace=True)
    return df

def evaluate_arima(series):
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    arima = ARIMA(train, order=(5,1,0))
    arima_fit = arima.fit()
    pred = arima_fit.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    mape = np.mean(np.abs((test - pred) / (test + 1e-8))) * 100
    accuracy = 100 - mape
    
    return accuracy

def evaluate_sarima(series):
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,24))
    sarima_fit = sarima.fit(disp=False)
    pred = sarima_fit.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    mape = np.mean(np.abs((test - pred) / (test + 1e-8))) * 100
    accuracy = 100 - mape
    
    return accuracy

def evaluate_lstm(series, steps=24, epochs=20, batch_size=32):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    
    X, y = [], []
    for i in range(len(scaled)-steps):
        X.append(scaled[i:i+steps])
        y.append(scaled[i+steps])
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / (y_test + 1e-8))) * 100
    accuracy = 100 - mape
    
    return accuracy

if __name__ == "__main__":
    file_path = "Air Quality.csv"
    df = load_and_preprocess(file_path)
    df = add_time_features(df)
    
    target = 'NO2(GT)'
    series = df[target]
    
    arima_acc = evaluate_arima(series)
    sarima_acc = evaluate_sarima(series)
    lstm_acc = evaluate_lstm(series)
    
    print(f"ARIMA Accuracy: {arima_acc:.2f}%")
    print(f"SARIMA Accuracy: {sarima_acc:.2f}%")
    print(f"LSTM Accuracy: {lstm_acc:.2f}%")

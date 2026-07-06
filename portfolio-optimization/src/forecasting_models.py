import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_test_split_chronological(df: pd.Series, split_date: str = "2025-01-01"):
    """
    Splits time series data chronologically to maintain temporal ordering.
    """
    train = df[:split_date].iloc[:-1] # Everything up to the day before split_date
    test = df[split_date:]           # Everything from split_date onwards
    return train, test

def fit_auto_arima(train_series: pd.Series):
    """
    Uses pmdarima's auto_arima to automatically find the optimal (p, d, q) parameters.
    """
    print("Running Auto-ARIMA parameter search... This might take a moment.")
    # We enforce d=1 since your ADF test confirmed differencing is required
    model = pm.auto_arima(train_series, start_p=1, start_q=1,
                          max_p=3, max_q=3, d=1, seasonal=False,
                          trace=True, error_action='ignore',  
                          suppress_warnings=True, stepwise=True)
    
    print(f"Optimal ARIMA parameters identified: {model.order}")
    return model

def calculate_forecast_metrics(y_true: pd.Series, y_pred: np.ndarray):
    """
    Calculates operational forecasting precision metrics: MAE, RMSE, and MAPE.
    Ensures safe array operations to avoid Datetime alignment mismatches.
    """
    # Force both vectors into flat numpy arrays
    true_vals = np.array(y_true).flatten()
    pred_vals = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    
    # Avoid zero division just in case
    epsilon = 1e-9
    mape = np.mean(np.abs((true_vals - pred_vals) / (true_vals + epsilon))) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": f"{mape:.2f}%"}


def prepare_lstm_sequences(train_series: pd.Series, test_series: pd.Series, window_size: int = 60):
    """
    Scales financial pricing sequences and reshapes data into [samples, time steps, features]
    windows for robust LSTM ingestion.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler on train data only to prevent look-ahead data leakage
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_series.values.reshape(-1, 1))
    
    # Combine back safely to build uninterrupted trailing sequences for the test set
    full_scaled = np.vstack((train_scaled, test_scaled))
    
    X_train, y_train = [], []
    for i in range(window_size, len(train_scaled)):
        X_train.append(train_scaled[i-window_size:i, 0])
        y_train.append(train_scaled[i, 0])
        
    X_test, y_test = [], []
    # Test windows look back into the late training sequence tail
    start_idx = len(train_scaled) - window_size
    for i in range(start_idx + window_size, len(full_scaled)):
        X_test.append(full_scaled[i-window_size:i, 0])
        y_test.append(full_scaled[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Reshape to 3D tensor: [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler

def build_and_train_lstm(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10, batch_size: int = 32):
    """
    Builds and optimizes a stacked LSTM architecture for deep financial time series regression.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("\nTraining Deep Learning LSTM network...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
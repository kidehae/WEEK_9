# import sys
# import os
# import numpy as np
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.eda_processor import load_and_clean_adj_close
# from src.forecasting_models import (
#     train_test_split_chronological, 
#     fit_auto_arima, 
#     calculate_forecast_metrics,
#     prepare_lstm_sequences,
#     build_and_train_lstm
# )

# def main():
#     raw_data_path = os.path.join("data", "processed", "historical_raw_data.csv")
    
#     # Load prices
#     prices = load_and_clean_adj_close(raw_data_path)
#     tsla_prices = prices['TSLA']
    
#     print(f"Total TSLA historical data points: {len(tsla_prices)}")
    
#     # 1. Split Data Chronologically
#     train, test = train_test_split_chronological(tsla_prices, split_date="2025-01-01")
#     print(f"Training points (2015-2024): {len(train)}")
#     print(f"Testing points (2025-2026): {len(test)}")
    
#     # 2. Fit optimal statistical model
#     fitted_auto_model = fit_auto_arima(train)
    
#     # 3. Generate predictions over the test space horizon length
#     forecast_steps = len(test)
#     predictions, conf_int = fitted_auto_model.predict(n_periods=forecast_steps, return_conf_int=True)
    
#     # 4. Evaluate metrics
#     metrics = calculate_forecast_metrics(test, predictions)
#     print("\n### ARIMA Model Evaluation Metrics on Unseen Test Window (2025-2026) ###")
#     for metric_name, value in metrics.items():
#         print(f"{metric_name}: {value}")

# if __name__ == "__main__":
#     main()


import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda_processor import load_and_clean_adj_close
from src.forecasting_models import (
    train_test_split_chronological, 
    fit_auto_arima, 
    calculate_forecast_metrics,
    prepare_lstm_sequences,
    build_and_train_lstm
)

def main():
    raw_data_path = os.path.join("data", "processed", "historical_raw_data.csv")
    prices = load_and_clean_adj_close(raw_data_path)
    tsla_prices = prices['TSLA']
    
    # 1. Split Data Chronologically
    train, test = train_test_split_chronological(tsla_prices, split_date="2025-01-01")
    
    # 2. Fit and Evaluate ARIMA
    fitted_auto_model = fit_auto_arima(train)
    arima_preds = fitted_auto_model.predict(n_periods=len(test))
    arima_metrics = calculate_forecast_metrics(test, arima_preds)
    
    # 3. Process Sequences and Fit LSTM
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_sequences(train, test, window_size=60)
    lstm_model = build_and_train_lstm(X_train, y_train, epochs=5, batch_size=32)
    
    # Generate LSTM predictions and reverse the scaling metric transform
    lstm_scaled_preds = lstm_model.predict(X_test)
    lstm_preds = scaler.inverse_transform(lstm_scaled_preds)
    lstm_metrics = calculate_forecast_metrics(test, lstm_preds)
    
    # 4. Final Head-to-Head Comparison Table
    print("\n" + "="*50)
    print("      TASK 2 MODEL COMPARISON REPORT")
    print("="*50)
    print(f"Metric   | ARIMA(0,1,0)   | Deep LSTM Network")
    print(f"-"*50)
    print(f"MAE      | {arima_metrics['MAE']:.4f}        | {lstm_metrics['MAE']:.4f}")
    print(f"RMSE     | {arima_metrics['RMSE']:.4f}        | {lstm_metrics['RMSE']:.4f}")
    print(f"MAPE     | {arima_metrics['MAPE']}         | {lstm_metrics['MAPE']}")
    print("="*50)

if __name__ == "__main__":
    main()
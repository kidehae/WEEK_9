import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Connect local src directory structures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import fetch_financial_data
from src.forecasting_models import train_lstm_model

def execute_future_trends():
    print("==================================================================")
    print("  TASK 3: INITIATING OUT-OF-SAMPLE ROLLING FUTURE PATH GENERATION ")
    print("==================================================================")
    
    # 1. Load clean historical asset datasets
    prices = fetch_financial_data()
    tsla_prices = prices['TSLA']
    
    # Chronological partition to replicate your model training configurations
    train_size = int(len(tsla_prices) * 0.85)
    train_series = tsla_prices.iloc[:train_size]
    
    print("\n[1/4] Training selected Stacked Long Short-Term Memory Predictive Model...")
    # Fits deep recurrent layers using look-back structural constraints
    model, scaler = train_lstm_model(train_series, window_size=60, epochs=3, batch_size=32)
    
    print("[2/4] Initializing recursive generation seed sequence from baseline data tail...")
    # Extract final 60 days of real observations ending June 30, 2026
    seed_window = tsla_prices.iloc[-60:].values
    current_input = scaler.transform(seed_window.reshape(-1, 1))
    
    future_forecasts = []
    forecast_horizon = 126  # 6 Calendar Months of standard forward trading sessions
    
    print(f"[3/4] Running multi-step recursive forecasting across {forecast_horizon} business steps...")
    for step in range(forecast_horizon):
        pred_input = np.reshape(current_input, (1, 60, 1))
        # Infer T+1 scaled valuation point
        pred_scaled = model.predict(pred_input, verbose=0)[0][0]
        future_forecasts.append(pred_scaled)
        
        # Shift look-back sequence vector forward: drop oldest, push new prediction to tail
        current_input = np.append(current_input[1:], [[pred_scaled]], axis=0)
        
    # Revert scale transformations back into nominal dollar market prices
    future_prices = scaler.inverse_transform(np.array(future_forecasts).reshape(-1, 1)).flatten()
    
    # 2. Uncertainty Ribbon calculations based on Task 2 Validation RMSE Bounds
    validation_rmse = 24.0625
    uncertainty_band = [validation_rmse * np.sqrt(i + 1) for i in range(forecast_horizon)]
    
    # Construct out-of-sample forward business calendar timeline dates
    future_dates = pd.date_range(start=tsla_prices.index[-1], periods=forecast_horizon + 1, freq='B')[1:]
    forecast_df = pd.Series(future_prices, index=future_dates)
    
    print("[4/4] Rendering visualization and exporting forecast data structures...")
    # Plot final predictive data outputs
    plt.figure(figsize=(12, 6))
    
    # Plot the trailing 250 sessions of actual historic observed prices for clarity
    plt.plot(tsla_prices.iloc[-250:], label="Historically Observed Market Prices", color="#1f77b4", linewidth=2)
    
    # Overlay the red recursive out-of-sample prediction curve
    plt.plot(forecast_df, label="6-Month Forward LSTM Projection", color="#d62728", linestyle="--", linewidth=2.5)
    
    # Shading the expanding confidence intervals/uncertainty bounds
    plt.fill_between(forecast_df.index, 
                     forecast_df.values - uncertainty_band, 
                     forecast_df.values + uncertainty_band, 
                     color="#ff9896", alpha=0.35, label='Expanding Structural Uncertainty Horizon (±RMSE*√t)')
    
    plt.title("GMF Asset Analytics: Out-of-Sample Rolling Future Path Inference (TSLA)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Timeline Horizon", fontsize=11)
    plt.ylabel("Asset Valuation Price ($)", fontsize=11)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e2e2e2")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Save chart visual output directly into the deliverables path
    plt.savefig('notebooks/visual4_future_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cache processed forward-looking files to feed directly into Task 4 Optimization
    output_df = pd.DataFrame(forecast_df, columns=['Predicted_Close'])
    output_df.to_csv('data/processed/tsla_future_forecast.csv')
    
    

if __name__ == "__main__":
    execute_future_trends()
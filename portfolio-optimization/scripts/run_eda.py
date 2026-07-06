import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda_processor import (
    load_and_clean_adj_close,
    calculate_returns_and_volatility,
    run_stationarity_test,
    calculate_risk_metrics,
    plot_and_save_all
)

def main():
    raw_data_path = os.path.join("data", "processed", "historical_raw_data.csv")
    
    # 1. Clean Data
    prices = load_and_clean_adj_close(raw_data_path)
    print("\n### Data Shapes and Sample ###")
    print(prices.tail(3))
    
    # 2. Calculate Returns
    returns, rolling_vol = calculate_returns_and_volatility(prices)
    
    # 3. Risk Metrics
    risk_summary = calculate_risk_metrics(returns)
    print("\n### Foundational Risk Metrics Summary ###")
    print(risk_summary)
    
    # 4. Stationarity Analysis (Price vs Returns for TSLA as required by Task 2)
    print("\n### Testing Stationarity (Prerequisite for ARIMA) ###")
    run_stationarity_test(prices['TSLA'], "TSLA Raw Price")
    run_stationarity_test(returns['TSLA'], "TSLA Daily Returns")

    # 5. Generate and Save Required Visualizations
    print("\n### Generating Visualizations... ###")
    plot_and_save_all(prices, returns, rolling_vol)

if __name__ == "__main__":
    main()
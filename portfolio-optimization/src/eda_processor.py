import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_and_clean_adj_close(file_path: str) -> pd.DataFrame:
    """
    Loads the raw data, extracts the Adjusted Close (or Close) prices for each asset,
    handles missing values, and ensures a proper datetime index.
    """
    # Read the file with multi-level columns
    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    
    adj_close_cols = {}
    
    # Let's inspect the columns to find the right price metric dynamically
    for col in df.columns:
        ticker, metric = col[0], col[1]
        
        # Look for 'Adj Close' or fallback to 'Close' if it's named differently
        if metric in ['Adj Close', 'Close']:
            if ticker not in adj_close_cols or metric == 'Adj Close':
                adj_close_cols[ticker] = df[col]
                
    # Build a clean DataFrame from our extracted columns
    adj_close_df = pd.DataFrame(adj_close_cols)
    
    # Handle missing values safely for financial time series
    adj_close_df = adj_close_df.ffill().bfill()
    
    return adj_close_df

def calculate_returns_and_volatility(df: pd.DataFrame, rolling_window: int = 20):
    """
    Calculates daily percentage returns and rolling volatility (standard deviation).
    """
    returns_df = df.pct_change().dropna()
    rolling_vol = returns_df.rolling(window=rolling_window).std() * np.sqrt(252) # Annualized
    return returns_df, rolling_vol

def run_stationarity_test(series: pd.Series, name: str):
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.
    """
    result = adfuller(series)
    print(f"\n--- Augmented Dickey-Fuller Test for {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4e}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    
    if result[1] <= 0.05:
        print(f"Result: Reject the null hypothesis (H0). The series is STATIONARY.")
    else:
        print(f"Result: Fail to reject the null hypothesis (H0). The series is NON-STATIONARY.")

def calculate_risk_metrics(returns_df: pd.DataFrame, confidence_level: float = 0.95):
    """
    Calculates foundational risk metrics: Value at Risk (VaR) and Historical Sharpe Ratio.
    """
    metrics = {}
    # Assuming risk-free rate is roughly 0 for daily metrics
    for col in returns_df.columns:
        # Historical VaR
        var_limit = np.percentile(returns_df[col], (1 - confidence_level) * 100)
        
        # Annualized Sharpe Ratio
        mean_return = returns_df[col].mean() * 252
        std_dev = returns_df[col].std() * np.sqrt(252)
        sharpe = mean_return / std_dev if std_dev != 0 else 0
        
        metrics[col] = {"95% Historical VaR": var_limit, "Annualized Sharpe Ratio": sharpe}
    return pd.DataFrame(metrics)

def plot_and_save_all(prices: pd.DataFrame, returns: pd.DataFrame, rolling_vol: pd.DataFrame):
    """
    Generates and saves the 3 required high-quality visualizations to the notebooks directory.
    """
    os.makedirs("notebooks", exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # Visual 1: Normalized Closing Prices (to compare trends on the same scale)
    plt.figure(figsize=(12, 6))
    normalized_prices = prices / prices.iloc[0] # Scale everything to start at 1.0
    for col in normalized_prices.columns:
        plt.plot(normalized_prices.index, normalized_prices[col], label=col)
    plt.title("Asset Growth Trends (Normalized to Base 1.0)", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/visual1_price_trends.png", dpi=300)
    plt.close()
    print("Saved: notebooks/visual1_price_trends.png")

    # Visual 2: Volatility Comparison (Rolling Standard Deviations)
    plt.figure(figsize=(12, 6))
    for col in rolling_vol.columns:
        plt.plot(rolling_vol.index, rolling_vol[col], label=f"{col} (20-Day Rolling Vol)")
    plt.title("Asset Volatility Comparison (Annualized Rolling Std Dev)", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/visual2_volatility.png", dpi=300)
    plt.close()
    print("Saved: notebooks/visual2_volatility.png")

    # Visual 3: Outlier and Return Distribution (Boxplot for Anomalies)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=returns)
    plt.title("Distribution of Daily Returns & Outlier Detection", fontsize=14, fontweight='bold')
    plt.ylabel("Daily Return Percentage")
    plt.xlabel("Assets")
    plt.tight_layout()
    plt.savefig("notebooks/visual3_outliers.png", dpi=300)
    plt.close()
    print("Saved: notebooks/visual3_outliers.png")
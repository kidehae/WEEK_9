import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Fix path resolution dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import fetch_financial_data, calculate_returns

def execute_portfolio_optimization():
    print("==================================================================")
    print("  TASK 4: INITIATING MODERN PORTFOLIO THEORY (MPT) OPTIMIZATION  ")
    print("==================================================================")
    
    # 1. Load historical and predicted data
    prices = fetch_financial_data()
    returns = calculate_returns(prices)
    
    # Read the LSTM forecast from Task 3 to extract our forward view return
    forecast_df = pd.read_csv('data/processed/tsla_future_forecast.csv', index_col=0, parse_dates=True)
    tsla_initial_price = prices['TSLA'].iloc[-1]
    tsla_forecasted_price = forecast_df['Predicted_Close'].iloc[-1]
    
    # Compute annualized expected return for TSLA based on the 6-month forecast horizon
    tsla_predicted_return = ((tsla_forecasted_price - tsla_initial_price) / tsla_initial_price) * 2  
    
    print(f"\n[1/4] Formulating expected returns vector...")
    # Calculate historical annualized baseline returns for BND and SPY
    mu = expected_returns.mean_historical_return(prices)
    # Inject our active alpha view for TSLA
    mu['TSLA'] = tsla_predicted_return
    
    print("Annualized Expected Returns Profile:")
    for asset, ret in mu.items():
        print(f"  * {asset}: {ret*100:.2f}%")
        
    print("\n[2/4] Generating robust Ledoit-Wolf shrinkage covariance matrix...")
    # Ledoit-Wolf shrinkage smooths extreme outlier shocks in high-volatility assets like TSLA
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    
    # Deliverable 1: Generate Covariance Correlation Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)
    plt.title("GMF Asset Allocation Universe: Correlation Matrix")
    plt.tight_layout()
    plt.savefig('notebooks/covariance_heatmap.png', dpi=300)
    plt.close()
    print("  -> Saved correlation heatmap to 'notebooks/covariance_heatmap.png'")

    print("\n[3/4] Running optimization simulations for key allocation targets...")
    # Optimize for Maximum Sharpe Ratio (Tangency Portfolio)
    ef_sharpe = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    raw_weights_sharpe = ef_sharpe.max_sharpe(risk_free_rate=0.04) # Assuming a 4% standard baseline risk-free rate
    cleaned_weights_sharpe = ef_sharpe.clean_weights()
    perf_sharpe = ef_sharpe.portfolio_performance(verbose=False, risk_free_rate=0.04)
    
    # Optimize for Minimum Volatility Portfolio
    ef_min_vol = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    raw_weights_min_vol = ef_min_vol.min_volatility()
    cleaned_weights_min_vol = ef_min_vol.clean_weights()
    perf_min_vol = ef_min_vol.portfolio_performance(verbose=False, risk_free_rate=0.04)
    
    # Deliverable 2: Log optimal metrics for the investment memo
    print("\n==========================================================")
    print("  FINAL PORTFOLIO RECOMMENDATION METRICS REPORT")
    print("==========================================================")
    print(f"A. MAXIMUM SHARPE RATIO PORTFOLIO (Tangency Optimization):")
    for asset, weight in cleaned_weights_sharpe.items():
        print(f"   - {asset} Target Allocation Weight: {weight*100:.2f}%")
    print(f"   * Expected Annual Return: {perf_sharpe[0]*100:.2f}%")
    print(f"   * Expected Portfolio Volatility: {perf_sharpe[1]*100:.2f}%")
    print(f"   * Optimized Sharpe Ratio: {perf_sharpe[2]:.4f}")
    
    print(f"\nB. MINIMUM VOLATILITY PORTFOLIO (Risk Mitigation Focus):")
    for asset, weight in cleaned_weights_min_vol.items():
        print(f"   - {asset} Target Allocation Weight: {weight*100:.2f}%")
    print(f"   * Expected Annual Return: {perf_min_vol[0]*100:.2f}%")
    print(f"   * Expected Portfolio Volatility: {perf_min_vol[1]*100:.2f}%")
    print(f"   * Optimized Sharpe Ratio: {perf_min_vol[2]:.4f}")
    print("==========================================================")
    
    # Cache optimization weight states for the Task 5 backtesting pipeline
    weights_df = pd.DataFrame([cleaned_weights_sharpe], index=['Max_Sharpe'])
    weights_df.to_csv('data/processed/optimized_portfolio_weights.csv')
    
    # 4. Deliverable 3: Simulate and Plot the Efficient Frontier Curve
    print("\n[4/4] Generating Efficient Frontier curve mapping...")
    plt.figure(figsize=(10, 6))
    
    # Create target points on the efficient frontier manually for scatter charting
    n_samples = 10000
    w_samples = np.random.dirichlet(np.ones(3), size=n_samples)
    
    # FIX: Use mu.values directly as an attribute instead of calling it as a function
    r_samples = np.dot(w_samples, mu.values)
    v_samples = np.zeros(n_samples)
    
    # Convert shrinkage matrix S to a numpy matrix for fast multiplication
    S_matrix = S.values if hasattr(S, 'values') else np.array(S)
    
    for i in range(n_samples):
        v_samples[i] = np.sqrt(np.dot(w_samples[i].T, np.dot(S_matrix, w_samples[i])))
        
    plt.scatter(v_samples * 100, r_samples * 100, c=(r_samples - 0.04)/v_samples, cmap='viridis', marker='o', s=1, alpha=0.3)
    
    # Plot our two anchor optimized portfolios distinctly
    plt.scatter(perf_sharpe[1]*100, perf_sharpe[0]*100, color='red', marker='*', s=200, label='Maximum Sharpe Portfolio')
    plt.scatter(perf_min_vol[1]*100, perf_min_vol[0]*100, color='blue', marker='X', s=150, label='Minimum Volatility Portfolio')
    
    plt.title("GMF Asset Management: MPT Efficient Frontier Charting", fontsize=12, fontweight='bold')
    plt.xlabel("Annualized Portfolio Volatility / Standard Deviation (%)", fontsize=10)
    plt.ylabel("Expected Annual Portfolio Returns (%)", fontsize=10)
    plt.colorbar(label='Sharpe Risk-Reward Ratio Spectrum')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('notebooks/efficient_frontier.png', dpi=300)
    plt.close()
    print("  -> Saved Efficient Frontier line to 'notebooks/efficient_frontier.png'")

if __name__ == "__main__":
    execute_portfolio_optimization()
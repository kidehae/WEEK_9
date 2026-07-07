import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import fetch_financial_data, calculate_returns

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculates the maximum peak-to-trough drawdown percentage."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def execute_backtesting():
    print("==================================================================")
    print("  TASK 5: INITIATING HISTORICAL STRATEGY BACKTESTING SIMULATION  ")
    print("==================================================================")
    
    # 1. Isolate the designated backtesting slice (Jan 2025 - Jan 2026)
    prices = fetch_financial_data()
    backtest_prices = prices.loc['2025-01-01':'2026-01-31']
    daily_returns = calculate_returns(backtest_prices)
    
    # 2. Define allocation weight weights
    # Benchmark: Static 60% SPY / 40% BND / 0% TSLA
    w_bench = np.array([0.00, 0.40, 0.60]) # Order matching column sequence: TSLA, BND, SPY
    
    # Strategy: Max Sharpe derived weights (0% TSLA, 0% BND, 100% SPY)
    w_strat = np.array([0.00, 0.00, 1.00])
    
    print("\n[1/3] Computing tracking cumulative return performance vectors...")
    # Calculate daily portfolio performance
    strat_daily = daily_returns.dot(w_strat)
    bench_daily = daily_returns.dot(w_bench)
    
    # Compounding returns into cumulative visual paths
    strat_cum = (1 + strat_daily).cumprod()
    bench_cum = (1 + bench_daily).cumprod()
    
    print("[2/3] Extracting comparative risk-adjusted backtest metrics...")
    # Total Returns
    total_ret_strat = strat_cum.iloc[-1] - 1
    total_ret_bench = bench_cum.iloc[-1] - 1
    
    # Annualized Returns (Approx 252 trading sessions per annum)
    ann_ret_strat = strat_daily.mean() * 252
    ann_ret_bench = bench_daily.mean() * 252
    
    # Annualized Volatility
    ann_vol_strat = strat_daily.std() * np.sqrt(252)
    ann_vol_bench = bench_daily.std() * np.sqrt(252)
    
    # Sharpe Ratios (Assuming a 4% risk-free benchmark baseline)
    rf = 0.04
    sharpe_strat = (ann_ret_strat - rf) / ann_vol_strat if ann_vol_strat > 0 else 0
    sharpe_bench = (ann_ret_bench - rf) / ann_vol_bench if ann_vol_bench > 0 else 0
    
    # Max Drawdowns
    max_dd_strat = calculate_max_drawdown(strat_cum)
    max_dd_bench = calculate_max_drawdown(bench_cum)
    
    print("\n==========================================================")
    print("  STRATEGY VS BENCHMARK BACKTEST PERFORMANCE RESULTS")
    print("==========================================================")
    print(f"METRIC            | STRATEGY (100% SPY) | BENCHMARK (60/40)")
    print(f"----------------------------------------------------------")
    print(f"Total Return      | {total_ret_strat*100:.2f}%             | {total_ret_bench*100:.2f}%")
    print(f"Annualized Return | {ann_ret_strat*100:.2f}%             | {ann_ret_bench*100:.2f}%")
    print(f"Sharpe Ratio      | {sharpe_strat:.4f}              | {sharpe_bench:.4f}")
    print(f"Max Drawdown      | {max_dd_strat*100:.2f}%            | {max_dd_bench*100:.2f}%")
    print("==========================================================")
    
    print("\n[3/3] Exporting performance charts...")
    plt.figure(figsize=(11, 5))
    plt.plot(strat_cum, label='Model-Optimized Strategy Portfolio (100% SPY)', color='#2ca02c', linewidth=2)
    plt.plot(bench_cum, label='Passive Balanced Benchmark Portfolio (60% SPY / 40% BND)', color='#7f7f7f', linestyle='--', linewidth=1.5)
    
    plt.title("GMF Investment Analytics: Strategy Backtesting Cumulative Return Performance Comparison", fontsize=11, fontweight='bold')
    plt.xlabel("Timeline Date Horizon", fontsize=9)
    plt.ylabel("Growth of Initial $1 Investment Base", fontsize=9)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('notebooks/backtest_cumulative_returns.png', dpi=300)
    plt.close()
    print("  -> Saved cumulative performance chart to 'notebooks/backtest_cumulative_returns.png'")

if __name__ == "__main__":
    execute_backtesting()
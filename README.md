# Asset Analytics & Portfolio Optimization Platform

## 10 Academy: Artificial Intelligence Mastery — Week 9 Challenge

**Project Duration:** July 1 – July 7, 2026  
**Lead Engineer:** Meaza Mulatu  
**Project Collaborators:** Mahlet Belay, Mekdelawit, Mery, Mariamawit  
**Client:** Guide Me in Finance (GMF) Investments

---

# Project Overview

This project develops an end-to-end quantitative investment analysis platform for **Guide Me in Finance (GMF) Investments**. It combines exploratory financial analytics, statistical forecasting, deep learning, and Modern Portfolio Theory (MPT) to build an intelligent asset allocation pipeline.

The system:

- Downloads historical market data directly from Yahoo Finance.
- Performs financial exploratory data analysis (EDA).
- Forecasts future asset prices using ARIMA and Deep LSTM models.
- Optimizes portfolio allocations using Modern Portfolio Theory.
- Backtests optimized investment strategies against benchmark portfolios.

The project analyzes three assets covering different market risk profiles over the period:

**January 1, 2015 – June 30, 2026**

| Asset | Description | Investment Role |
|--------|-------------|-----------------|
| **TSLA** | Tesla Inc. | High-growth, high-volatility equity (Alpha generation) |
| **SPY** | SPDR S&P 500 ETF | Broad-market diversified equity exposure |
| **BND** | Vanguard Total Bond Market ETF | Investment-grade bond ETF for stability and downside protection |

---

# Project Objectives

The project is organized into five major analytical stages:

1. Data Collection & Exploratory Data Analysis
2. Time Series Forecasting
3. Future Market Trend Prediction
4. Portfolio Optimization
5. Portfolio Strategy Backtesting

---

# Repository Structure

```text
portfolio-optimization/
│
├── .github/
│   └── workflows/
│       └── unittests.yml
│
├── .vscode/
│   └── settings.json
│
├── data/
│   └── processed/
│       ├── historical_raw_data.csv
│       ├── tsla_future_forecast.csv
│       └── optimized_portfolio_weights.csv
│
├── notebooks/
│   ├── README.md
│   ├── Interim_Report.md
│   ├── visual1_price_trends.png
│   ├── visual2_volatility.png
│   ├── visual3_outliers.png
│   ├── visual4_future_forecast.png
│   ├── covariance_heatmap.png
│   ├── efficient_frontier.png
│   └── backtest_cumulative_returns.png
│
├── scripts/
│   ├── run_eda.py
│   ├── run_forecasting.py
│   ├── run_trends_forecasting.py
│   ├── run_portfolio_optimization.py
│   └── run_strategy_backtesting.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda_processor.py
│   └── forecasting_models.py
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python 3.11+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- TensorFlow / Keras
- PyPortfolioOpt
- yfinance
- scikit-learn
- pmdarima

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/portfolio-optimization.git
cd portfolio-optimization
```

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Git Bash / Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Running the Project

Run all commands from the project root directory.

---

## Task 1 — Exploratory Data Analysis

Runs:

- Data download
- Cleaning
- Missing value handling
- Rolling volatility
- Value at Risk (VaR)
- Sharpe Ratios
- ADF Stationarity Test
- Data visualization

```bash
python scripts/run_eda.py
```

Outputs include:

- Price trend visualization
- Rolling volatility
- Outlier analysis
- Risk statistics

---

## Task 2 — Forecast Validation Models

Builds and evaluates:

- Auto ARIMA
- Deep Stacked LSTM

Evaluation metrics:

- MAE
- RMSE
- MAPE

Chronological split:

- Training: 2015–2024
- Testing: 2025–2026

```bash
python scripts/run_forecasting.py
```

---

## Task 3 — Future Trend Forecasting

Uses the trained LSTM model to recursively forecast approximately six months of future Tesla prices.

Generates:

- Future forecast CSV
- Forecast visualization
- Confidence intervals

```bash
python scripts/run_trends_forecasting.py
```

---

## Task 4 — Portfolio Optimization

Uses historical returns together with forecasted expectations to compute optimal portfolio allocations.

The optimization pipeline:

- Expected Returns
- Ledoit-Wolf Covariance Matrix
- Efficient Frontier
- Maximum Sharpe Portfolio
- Minimum Volatility Portfolio

```bash
python scripts/run_portfolio_optimization.py
```

Generated outputs:

- Efficient Frontier
- Covariance Heatmap
- Portfolio Weights CSV

---

## Task 5 — Strategy Backtesting

Compares the optimized portfolio against a passive benchmark.

Backtesting period:

**January 2025 – January 2026**

Performance metrics include:

- Total Return
- Annualized Return
- Sharpe Ratio
- Maximum Drawdown

```bash
python scripts/run_strategy_backtesting.py
```

---

# Key Analytical Findings

## Exploratory Data Analysis

### Historical Value at Risk (95%)

TSLA exhibited a historical one-day Value at Risk of approximately:

**−5.17%**

This indicates that on only 5% of historical trading days Tesla lost more than 5.17%.

---

### Sharpe Ratio

SPY achieved a superior risk-adjusted return relative to TSLA, demonstrating the diversification benefits of broad-market exposure.

---

### Stationarity (ADF Test)

Raw TSLA prices:

- p ≈ 0.727
- Non-stationary

Differenced returns:

- p ≈ 0.000
- Stationary

This validates using differencing (d = 1) in ARIMA.

---

# Forecasting Performance

| Metric | ARIMA | Deep LSTM |
|---------|------:|----------:|
| MAE | 58.75 | **19.01** |
| RMSE | 77.51 | **24.06** |
| MAPE | 18.88% | **5.14%** |

The Deep LSTM substantially outperformed the statistical baseline across every evaluation metric.

---

# Future Market Forecast

The trained LSTM projected a significant downward trend in Tesla over the forecast horizon.

Estimated annualized expected return used during optimization:

**−40.97%**

Forecast uncertainty widened over time, consistent with the Efficient Market Hypothesis.

---

# Portfolio Optimization Results

## Maximum Sharpe Portfolio

| Asset | Allocation |
|--------|-----------:|
| TSLA | 0.00% |
| SPY | 100.00% |
| BND | 0.00% |

Expected Return

**13.73%**

Portfolio Volatility

**17.75%**

Sharpe Ratio

**0.5481**

---

## Minimum Volatility Portfolio

| Asset | Allocation |
|--------|-----------:|
| TSLA | 0.00% |
| SPY | 6.79% |
| BND | 93.21% |

Expected Return

**2.68%**

Portfolio Volatility

**5.61%**

---

# Strategy Backtesting Results

| Metric | Active Strategy | Passive 60/40 Benchmark |
|--------|----------------:|-------------------------:|
| Total Return | **19.75%** | 15.11% |
| Annualized Return | **18.66%** | 13.87% |
| Sharpe Ratio | 0.7726 | **0.8439** |
| Maximum Drawdown | -18.76% | **-11.29%** |

The optimized active allocation generated higher returns than the benchmark, although the concentrated equity exposure resulted in larger drawdowns.

---

# Visual Outputs

The project generates several figures automatically:

- Asset Price Trends
- Rolling Volatility
- Outlier Analysis
- Future Forecast
- Covariance Heatmap
- Efficient Frontier
- Portfolio Backtest Performance

All figures are saved under the `notebooks/` directory.

---

# Conclusion

This project demonstrates a complete quantitative investment workflow integrating:

- Financial data engineering
- Exploratory data analysis
- Statistical time-series forecasting
- Deep learning with LSTM networks
- Modern Portfolio Theory optimization
- Portfolio strategy backtesting

The resulting framework enables data-driven investment decisions by combining predictive analytics with quantitative portfolio construction techniques, providing a robust foundation for intelligent asset allocation.

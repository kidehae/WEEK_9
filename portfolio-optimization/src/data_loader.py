# import yfinance as yf
# import pandas as pd
# import os
# from typing import List

# def fetch_asset_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
#     """
#     Fetches historical financial data from YFinance for a list of tickers.
    
#     Parameters:
#         tickers (List[str]): List of asset symbols (e.g., ['TSLA', 'BND', 'SPY'])
#         start_date (str): Start date string in 'YYYY-MM-DD' format
#         end_date (str): End date string in 'YYYY-MM-DD' format
        
#     Returns:
#         pd.DataFrame: Long-form multi-indexed dataframe containing historical data.
#     """
#     print(f"Initiating download for tickers: {tickers} from {start_date} to {end_date}...")
#     try:
#         # Fetching data with group_by='ticker' to easily separate out metrics later
#         data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
#         if data.empty:
#             raise ValueError("The downloaded DataFrame is empty. Check your tickers or date range.")
#         return data
#     except Exception as e:
#         print(f"An error occurred during data extraction: {e}")
#         raise e

# def save_to_processed(df: pd.DataFrame, filename: str) -> None:
#     """Saves the structured DataFrame safely to the data/processed folder."""
#     output_dir = os.path.join("data", "processed")
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, filename)
#     df.to_csv(output_path)
#     print(f"Data successfully serialized and saved to {output_path}")

# if __name__ == "__main__":
#     # Test our pipeline locally using the dates defined in your challenge document
#     TICKERS = ["TSLA", "BND", "SPY"]
#     START = "2015-01-01"
#     END = "2026-06-30"
    
#     raw_data = fetch_asset_data(tickers=TICKERS, start_date=START, end_date=END)
#     save_to_processed(raw_data, "historical_raw_data.csv")


import yfinance as yf
import pandas as pd
import os
from typing import List

def fetch_financial_data(tickers: List[str] = ["TSLA", "BND", "SPY"], start_date: str = "2015-01-01", end_date: str = "2026-06-30") -> pd.DataFrame:
    """
    Fetches historical financial data from YFinance for a list of tickers, Isolating Adj Close pricing safely.
    """
    print(f"Initiating download for tickers: {tickers} from {start_date} to {end_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("The downloaded DataFrame is empty. Check your tickers or date range.")
        
        # --- Robust Column Picker Integration ---
        # Look for standard MultiIndex levels or flat column headers flexibly
        adj_close_data = pd.DataFrame()
        
        for ticker in tickers:
            # Check for MultiIndex setups: (Metric, Ticker) or (Ticker, Metric)
            if isinstance(data.columns, pd.MultiIndex):
                if ('Adj Close', ticker) in data.columns:
                    adj_close_data[ticker] = data[('Adj Close', ticker)]
                elif ('adj close', ticker) in data.columns:
                    adj_close_data[ticker] = data[('adj close', ticker)]
                elif ('Close', ticker) in data.columns:
                    adj_close_data[ticker] = data[('Close', ticker)]
                else:
                    # Fallback lookup by locating key components inside the tuples
                    found = False
                    for col in data.columns:
                        if col[1] == ticker and 'close' in col[0].lower():
                            adj_close_data[ticker] = data[col]
                            found = True
                            break
                    if not found:
                        raise KeyError(f"Could not locate closing metrics for {ticker}")
            else:
                # Flat index fallback (Single asset download shapes)
                if 'Adj Close' in data.columns:
                    adj_close_data = data[['Adj Close']]
                elif 'Close' in data.columns:
                    adj_close_data = data[['Close']]
                break

        # Imputation: forward-fill followed by backward-fill to protect timeline integrity
        adj_close_data = adj_close_data.ffill().bfill()
        return adj_close_data
        
    except Exception as e:
        print(f"An error occurred during data extraction: {e}")
        raise e
def save_to_processed(df: pd.DataFrame, filename: str) -> None:
    """Saves the structured DataFrame safely to the data/processed folder."""
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path)
    print(f"Data successfully serialized and saved to {output_path}")

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily percentage returns."""
    return df.pct_change().dropna()
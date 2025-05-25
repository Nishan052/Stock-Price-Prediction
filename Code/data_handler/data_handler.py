import pandas as pd
import numpy as np
import yfinance as yf
from  config import get_last_weekday

def load_nifty50_yfinance(start="2010-01-01", end=None):
    """
    Loads historical NIFTY 50 index data using yfinance,
    cleans it, filters weekdays, and adds a COVID dummy variable.

    Returns all relevant numeric columns needed by ARIMA/LSTM models.
    """
    # Download NIFTY 50 index data
    if end is None:
        end = get_last_weekday()
    nifty = yf.Ticker("^NSEI")
    df = nifty.history(start=start, end=end, interval="1d")

    # Ensure 'Date' is the index name
    df.index.name = 'Date'

    # Keep only weekdays
    df = df[df.index.dayofweek < 5]

    # Add COVID dummy variable (1 during March–Dec 2020)
    df["COVID_dummy"] = 0
    df.loc["2020-03-01":"2020-12-31", "COVID_dummy"] = 1

    # Drop rows where 'Close' is missing
    df.dropna(subset=["Close"], inplace=True)

    # Select only relevant columns if they exist
    desired_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    available_cols = [col for col in desired_cols if col in df.columns]
    df = df[available_cols + ["COVID_dummy"]]

    # Sort by date
    df.sort_index(inplace=True)

    return df
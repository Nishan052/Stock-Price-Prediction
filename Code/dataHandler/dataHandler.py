##
# @file dataHandler.py
# @brief Loads and preprocesses NIFTY 50 stock market data from Yahoo Finance.
#
# @details
# This module fetches historical data via `yfinance`, filters valid trading days,
# handles missing values, injects a COVID-19 dummy variable for modeling exogenous effects,
# computes z-scores for outlier detection, and logs identified outliers.
# It returns a cleaned DataFrame ready for training or forecasting.
#
# @date June 2025

import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
import logging

baseDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(baseDir, "..")))

from utils.messageHandler import MessageHandler
from utils.errorHandler import logError

msg = MessageHandler()

##
# @brief Loads and processes NIFTY 50 index data using the `yfinance` API.
#
# @param start The start date for data download. Default is "2008-01-01".
# @param end The end date for data download. Default is "2025-01-01".
# @return pd.DataFrame Time-indexed DataFrame with cleaned and enriched features, including z-score and COVID dummy.
#
# @details
# Processing steps:
# - Download daily price data
# - Filter non-trading days (weekends)
# - Drop rows with missing 'Close' values
# - Add COVID dummy variable for March–Dec 2020
# - Compute z-score for 'Close' and identify outliers (|zScore_Close| > 3)
# - Retain only essential stock columns
# - Ensure chronological order
# - Log number of outliers detected
##
def loadNifty50Yfinance(start="2008-01-01", end="2025-01-01"):
    try:
        logging.info(msg.get("loading_nifty_data"))

        # Download NIFTY 50 data from Yahoo Finance
        nifty = yf.Ticker("^NSEI")
        df = nifty.history(start=start, end=end, interval="1d")
        logging.info("Data downloaded successfully.")

        # Ensure datetime index is labeled
        df.index.name = 'Date'

        # Keep only weekdays (Monday to Friday)
        logging.info(msg.get("filtering_weekdays"))
        df = df[df.index.dayofweek < 5]

        # Add COVID dummy variable: 1 for March–Dec 2020, 0 otherwise
        logging.info(msg.get("adding_covid_dummy"))
        df["COVID_dummy"] = 0
        df.loc["2020-03-01":"2020-12-31", "COVID_dummy"] = 1

        # Drop rows with missing 'Close' values
        logging.info(msg.get("dropping_na_close"))
        initial_rows = len(df)
        df.dropna(subset=["Close"], inplace=True)
        dropped_rows = initial_rows - len(df)
        logging.info(f"Dropped {dropped_rows} rows with missing 'Close' values.")

        # Compute z-score for 'Close' column
        logging.info("Calculating z-score for 'Close' values.")
        mean_close = df['Close'].mean()
        std_close = df['Close'].std()
        df['zScore_Close'] = (df['Close'] - mean_close) / std_close

        # Identify outliers where absolute z-score > 3
        outliers = df[np.abs(df['zScore_Close']) > 3]
        num_outliers = len(outliers)
        logging.info(f"Detected {num_outliers} outliers with |zScore_Close| > 3.")

        # Optionally, print or log the details of outliers
        if num_outliers > 0:
            logging.debug("Outlier details:\n%s", outliers[['Close', 'zScore_Close']].to_string())

        # Select only relevant columns for modeling
        logging.info(msg.get("selecting_columns"))
        desiredCols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
        availableCols = [col for col in desiredCols if col in df.columns]
        df = df[availableCols + ["COVID_dummy", "zScore_Close"]]

        # Sort data chronologically to maintain temporal consistency
        logging.info(msg.get("sorting_by_date"))
        df.sort_index(inplace=True)

        logging.info("NIFTY 50 data loaded and processed successfully.")
        return df

    except Exception as e:
        logError(e, context="load_nifty50_yfinance")
        raise
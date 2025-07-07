##
# @file test_data_handler.py
# @brief Unit tests for data loading and preprocessing in `data_handler.py`.
#
# @details
# This test module verifies weekday filtering, NaN handling, COVID dummy flagging,
# column selection, and proper index sorting for historical NIFTY 50 data loaded using `yfinance`.
#
# @date June 2025

##

import pandas as pd
import numpy as np
import pytest

from dataHandler.dataHandler import loadNifty50Yfinance

##
# @class DummyTicker
# @brief A dummy mock class to simulate `yfinance.Ticker` responses.
#
# @details
# Simulates historical NIFTY 50 data for test purposes. Injects a missing value
# and unordered dates to validate preprocessing steps such as NaN removal and index sorting.
##
class DummyTicker:
    def __init__(self, symbol):
        pass

    ##
    # @brief Returns mock historical data for testing.
    #
    # @param start Start date string (unused here)
    # @param end End date string (unused here)
    # @param interval Time interval string (unused here)
    # @return pd.DataFrame Simulated stock data with some edge cases
    ##
    def history(self, start, end, interval):
        # Include weekend and a NaN in 'Close' on 2020-03-02
        dates = pd.date_range("2020-02-27", "2020-03-05", freq="D")
        df = pd.DataFrame({
            "Open":         np.arange(len(dates)),
            "High":         np.arange(len(dates)) + 1,
            "Low":          np.arange(len(dates)) - 1,
            "Close":        np.arange(len(dates)) + 2,
            "Volume":       np.arange(len(dates)) * 100,
            "Dividends":    np.zeros(len(dates)),
            "Stock Splits": np.zeros(len(dates)),
            "Adj Close":    np.arange(len(dates)) + 3,
        }, index=dates)
        df.index.name = "Date"
        df.loc["2020-03-02", "Close"] = np.nan  # Inject missing value
        return df.sample(frac=1)  # Shuffle rows to test sorting

##
# @brief Fixture to patch yfinance.Ticker with DummyTicker automatically for all tests.
#
# @param monkeypatch Pytest fixture to monkey-patch objects
##
@pytest.fixture(autouse=True)
def patchYf(monkeypatch):
    import dataHandler.dataHandler as dh
    monkeypatch.setattr(dh.yf, "Ticker", DummyTicker)

##
# @test
# @brief Validates that only weekdays are kept and NaNs are dropped.
#
# @details
# Ensures that the 'Close' NaN is removed and dates are sorted correctly.
##
def testWeekdaysFilteredAndNanDropped():
    df = loadNifty50Yfinance(start="2020-02-27", end="2020-03-06")
    assert all(df.index.dayofweek < 5)  # Only weekdays (Mon–Fri)
    assert "2020-03-02" not in df.index.astype(str)  # NaN was removed
    assert df.index.is_monotonic_increasing  # Dates are sorted

##
# @test
# @brief Verifies the correct assignment of COVID dummy variable.
#
# @details
# Ensures 0 for dates before March 2020 and 1 for March–December 2020.
##
def testCovidDummyFlag():
    df = loadNifty50Yfinance(start="2020-02-27", end="2020-04-01")
     
    assert (df.loc[: "2020-02-29", "COVID_dummy"] == 0).all()
    df = df.sort_index()
    assert (df.loc["2020-03-01" : "2020-03-31", "COVID_dummy"] == 1).all()

##
# @test
# @brief Checks that only expected columns are retained in the dataset.
#
# @details
# Validates that the final DataFrame includes exactly the predefined set of columns.
##
def testColumnsSelectedExactly():
    df = loadNifty50Yfinance(start="2020-02-27", end="2020-03-06")
    expected = {
        "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "COVID_dummy", "zScore_Close"
    }
    assert set(df.columns) == expected
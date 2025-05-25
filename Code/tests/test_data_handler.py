import pandas as pd
import numpy as np
import pytest

from data_handler.data_handler import load_nifty50_yfinance

class DummyTicker:
    def __init__(self, symbol):
        pass

    def history(self, start, end, interval):
        # date range including a weekend and a NaN on 3/2/2020
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
        # inject a NaN
        df.loc["2020-03-02", "Close"] = np.nan
        # shuffle to test sort_index
        return df.sample(frac=1)

@pytest.fixture(autouse=True)
def patch_yf(monkeypatch):
    import data_handler.data_handler as dh
    monkeypatch.setattr(dh.yf, "Ticker", DummyTicker)

def test_weekdays_filtered_and_nan_dropped():
    df = load_nifty50_yfinance(start="2020-02-27", end="2020-03-06")
    # only weekdays
    assert all(df.index.dayofweek < 5)
    # NaN on 2020-03-02 is dropped
    assert "2020-03-02" not in df.index.astype(str)
    # index is sorted
    assert df.index.is_monotonic_increasing

def test_covid_dummy_flag():
    df = load_nifty50_yfinance(start="2020-02-27", end="2020-04-01")
    # before March → 0
    assert (df.loc[: "2020-02-29", "COVID_dummy"] == 0).all()
    # March onward → 1
    assert (df.loc["2020-03-01" : "2020-03-31", "COVID_dummy"] == 1).all()

def test_columns_selected_exactly():
    df = load_nifty50_yfinance(start="2020-02-27", end="2020-03-06")
    expected = {
        "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "COVID_dummy"
    }
    assert set(df.columns) == expected
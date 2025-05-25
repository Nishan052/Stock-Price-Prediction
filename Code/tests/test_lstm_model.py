import pandas as pd
import numpy as np

import models.lstm.lstm_model as lstm_mod
from models.lstm.lstm_model import run_lstm

def test_run_lstm_all_nan_on_insufficient_data(monkeypatch):
    # 1) Tiny DataFrame, far too short for LOOKBACK=60
    idx = pd.date_range("2022-01-01", "2022-01-10", freq="D")
    df = pd.DataFrame({
        "Open":         np.arange(len(idx)),
        "Close":        np.arange(len(idx)) + 0.5,
        "Volume":       np.ones(len(idx)),
        "Dividends":    np.zeros(len(idx)),
        "Stock Splits": np.zeros(len(idx)),
        "COVID_dummy":  np.zeros(len(idx)),
    }, index=idx)

    # 2) Force TRAIN_END_DATE way before so that test_index = all days
    monkeypatch.setattr(lstm_mod, "get_train_end_date",
                        lambda df: df.index.max() - pd.DateOffset(years=2))
    monkeypatch.setattr(lstm_mod, "ROLLING_WINDOW_YEARS", 1)
    monkeypatch.setattr(lstm_mod, "LOOKBACK", 60)
    monkeypatch.setattr(lstm_mod, "RETRAIN_INTERVAL", 5)

    preds = run_lstm(df, columns=["Open", "Close"])
    # keys
    assert set(preds.keys()) == {"Open", "Close"}

    # each series is same length as idx and entirely NaN
    for col, ser in preds.items():
        assert isinstance(ser, pd.Series)
        assert len(ser) == len(idx)
        assert ser.isna().all()
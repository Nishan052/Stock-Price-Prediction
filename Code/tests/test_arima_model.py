import os
import numpy as np
import pandas as pd
import pytest

import models.arima.arima_model as arima_mod

def test_run_arima_with_dummy_models(tmp_path, monkeypatch):
    # 1) Build a tiny DataFrame
    idx = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    df = pd.DataFrame({
        "Open":  np.arange(len(idx)),
        "Close": np.arange(len(idx)) + 0.5
    }, index=idx)

    # 2) Force TRAIN_END_DATE = 2020-01-05
    monkeypatch.setattr(arima_mod, "get_train_end_date", lambda df: pd.Timestamp("2020-01-05"))
    # 3) Make rolling window 1 year
    monkeypatch.setattr(arima_mod, "ROLLING_WINDOW_YEARS", 1)

    # 4) Stub out auto_arima → object with .order
    class DummyAuto:
        order = (1, 0, 0)
    monkeypatch.setattr(arima_mod, "auto_arima", lambda y, **kw: DummyAuto())

    # 5) Stub out ARIMA so that .fit() returns an object with .forecast()
    class DummyModel:
        def forecast(self, steps):
            return pd.Series([42], index=[0])
    class DummyARIMA:
        def __init__(self, y, order):
            pass
        def fit(self):
            return DummyModel()
    monkeypatch.setattr(arima_mod, "ARIMA", DummyARIMA)

    # 6) Redirect MODEL_DIR to temp
    monkeypatch.setattr(arima_mod, "MODEL_DIR", str(tmp_path))

    # 7) Spy on joblib.dump
    saved = []
    monkeypatch.setattr(arima_mod.joblib, "dump", lambda mdl, path: saved.append(path))

    # Run
    out = arima_mod.run_arima(df, columns=["Open"], retrain=True)

    # –– Assertions
    assert isinstance(out, dict)
    assert "Open" in out
    series = out["Open"]
    # days > 2020-01-05 are Jan 6…Jan 10 → 5 forecasts
    assert len(series) == 5
    # every value is our dummy “42”
    assert (series == 42).all()

    # confirm we saved exactly one model file
    assert len(saved) == 1
    assert saved[0].endswith("arima_model_open.pkl")
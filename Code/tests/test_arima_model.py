##
# @file test_arima_model.py
# @brief Unit test for ARIMA walk-forward model using monkeypatching and dummy objects.
#
# @details
# This test validates the behavior of `run_arima()` without training real models.
# It uses dummy ARIMA and autoArima classes, forces the training/test split,
# and asserts correct structure and saving of model predictions.
#
# @date June 2025

##

import os
import numpy as np
import pandas as pd
import pytest

import models.arima.arimaModel as arimaMod

##
# @brief Test ARIMA training pipeline using dummy/stubbed dependencies.
#
# @param tmp_path Pytest fixture providing a temporary directory for model output.
# @param monkeypatch Pytest fixture to dynamically override module attributes and classes.
#
# @details
# Steps:
# 1. Create dummy time series data
# 2. Monkeypatch internal config (train end date, window size)
# 3. Stub autoArima and ARIMA to simulate model behavior
# 4. Intercept model saving
# 5. Run ARIMA forecasting and validate outputs
##
def testRunArimaWithDummyModels(tmp_path, monkeypatch):
    ## 1. Generate dummy DataFrame for dates Jan 1–10, 2020
    idx = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    df = pd.DataFrame({
        "Open":  np.arange(len(idx)),
        "Close": np.arange(len(idx)) + 0.5
    }, index=idx)

    ## 2. Force training cutoff at Jan 5, 2020
    monkeypatch.setattr(arimaMod, "getTrainEndDate", lambda df: pd.Timestamp("2020-01-05"))

    ## 3. Set rolling window to 1 year
    monkeypatch.setattr(arimaMod, "rollingWindowYears", 1)

    ##
    # 4. Dummy autoArima object that mimics output with a fixed order
    ##
    class DummyAuto:
        order = (1, 0, 0)
    monkeypatch.setattr(arimaMod, "auto_arima", lambda y, **kw: DummyAuto())

    ##
    # 5. Stub ARIMA.fit() to return a DummyModel with fixed forecast output
    ##
    class DummyModel:
        def forecast(self, steps):
            return pd.Series([42], index=[0])

    class DummyARIMA:
        def __init__(self, y, order):
            pass
        def fit(self):
            return DummyModel()

    monkeypatch.setattr(arimaMod, "ARIMA", DummyARIMA)

    ## 6. Redirect modelDir to temp folder to prevent filesystem pollution
    monkeypatch.setattr(arimaMod, "modelDir", str(tmp_path))

    ##
    # 7. Replace joblib.dump with a lambda that tracks saved file paths
    ##
    saved = []
    monkeypatch.setattr(arimaMod.joblib, "dump", lambda mdl, path: saved.append(path))

    ## Run ARIMA walk-forward forecast
    out = arimaMod.runArima(df, columns=["Open"], retrain=True)

    ##
    # Validate returned forecast structure and contents
    ##
    assert isinstance(out, dict)
    assert "Open" in out
    series = out["Open"]

    ## Dates after Jan 5 = Jan 6 to Jan 10 → 5 forecast steps
    assert len(series) == 5

    ## All dummy values should equal 42
    assert (series == 42).all()

    ## Ensure only one model was saved
    assert len(saved) == 1
    assert saved[0].endswith("arimaModelOpen.pkl")
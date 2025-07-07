##
# @file test_lstm_model.py
# @brief Unit test for `runLstm()` to ensure it returns NaNs when insufficient data is available.
#
# @details
# This test checks how the LSTM model behaves when the DataFrame contains fewer rows
# than the required `lookback` value (60 days). All forecast outputs are expected to be NaN.
#
# @date June 2025
# @section author Author

##

import pandas as pd
import numpy as np

import models.lstm.lstmModel as lstm_mod
from models.lstm.lstmModel import runLstm

##
# @test
# @brief Tests that `runLstm()` returns all NaNs when the dataset is too small for training.
#
# @details
# Mocks the required configuration to ensure the `lookback` is 60 while providing only 10 days of data.
# It checks that predictions for both "Open" and "Close" are Series of NaNs, matching the length of the test period.
#
# @param monkeypatch Pytest fixture to mock constants and function calls inside `lstmModel`.
##
def testRunLstmAllAanOnInsufficientData(monkeypatch):
    # 1) Create a DataFrame with just 10 days (less than lookback=60)
    idx = pd.date_range("2022-01-01", "2022-01-10", freq="D")
    df = pd.DataFrame({
        "Open":         np.arange(len(idx)),
        "Close":        np.arange(len(idx)) + 0.5,
        "Volume":       np.ones(len(idx)),
        "Dividends":    np.zeros(len(idx)),
        "Stock Splits": np.zeros(len(idx)),
        "COVID_dummy":  np.zeros(len(idx)),
    }, index=idx)

    # 2) Force training cutoff to make all dates fall into the test set
    monkeypatch.setattr(lstm_mod, "getTrainEndDate",
                        lambda df: df.index.max() - pd.DateOffset(years=2))

    # 3) Override key parameters to match the actual model setup
    monkeypatch.setattr(lstm_mod, "rollingWindowYears", 1)
    monkeypatch.setattr(lstm_mod, "lookback", 60)          # Minimum history required
    monkeypatch.setattr(lstm_mod, "retrainInterval", 5)   # Retrain every 5 steps

    # 4) Run the model with the dummy DataFrame
    preds = runLstm(df, columns=["Open", "Close"])

    # 5) Ensure predictions contain keys for both target columns
    assert set(preds.keys()) == {"Open", "Close"}

    # 6) Check that every prediction is a Series of NaNs, and matches index length
    for col, ser in preds.items():
        assert isinstance(ser, pd.Series)
        assert len(ser) == len(idx)
        assert ser.isna().all()
##
# @file test_main.py
# @brief Integration test to simulate full pipeline execution from `main.py`.
#
# @details
# This test mocks the data loader, ARIMA model, LSTM model, and matplotlib plotting components.
# It ensures `main.py` executes end-to-end without depending on external I/O, model training, or real plotting.
# The test also verifies that the terminal output contains the expected forecast message.
#
# @date June 2025

##

import os
import pandas as pd
import pytest
from unittest.mock import MagicMock

##
# @test
# @brief End-to-end test simulating the entire `main.py` pipeline using monkeypatch.
#
# @param monkeypatch Pytest fixture used to override and simulate dependencies.
# @param tmpPath Temporary path to isolate file writes.
# @param capsys Pytest fixture to capture terminal output.
##
##
# @brief End-to-end integration test for the main NIFTY 50 forecasting pipeline.
#
# @details
# This test mocks external dependencies (data loading, models, plotting) and executes the `main.py` module as if it were run normally.
# It uses monkeypatching to simulate controlled input/output and verifies that the forecast summary is printed to the terminal.
#
# @param monkeypatch Pytest fixture for replacing or modifying functions and objects at runtime.
# @param tmpPath Temporary directory fixture used to isolate file writes and avoid polluting the working directory.
# @param capsys Pytest fixture to capture standard output and error for validation.
##
def testMainEndToEnd(monkeypatch, tmp_path, capsys):
    ##
    # @brief Create a mock DataFrame with two dates and minimal data.
    #
    # @details Simulates historical stock data for NIFTY 50 with "Open" and "Close" prices.
    ##
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    stub_df = pd.DataFrame({"Open": [1, 2], "Close": [1.1, 2.2]}, index=dates)

    ##
    # @brief Import actual modules to monkeypatch target functions.
    ##
    import dataHandler.dataHandler as dh
    import models.arima.arimaModel as ar_mod
    import models.lstm.lstmModel as lstm_mod

    ##
    # @brief Replace data loading with mock DataFrame.
    ##
    monkeypatch.setattr(dh, "loadNifty50Yfinance", lambda *a, **k: stub_df)

    ##
    # @brief Provide static prediction outputs for ARIMA and LSTM models.
    #
    # @details These mock predictions simulate actual model behavior for testing.
    ##
    ar_preds = {
        "Open": pd.Series([10, 20], index=dates),
        "Close": pd.Series([11, 21], index=dates)
    }
    ls_preds = {
        "Open": pd.Series([12, 22], index=dates),
        "Close": pd.Series([13, 23], index=dates)
    }

    monkeypatch.setattr(ar_mod, "runArima", lambda df, columns=None: ar_preds)
    monkeypatch.setattr(lstm_mod, "runLstm", lambda df, columns=None: ls_preds)

    ##
    # @brief Mock all matplotlib plotting calls to suppress actual plotting.
    #
    # @details Prevents display or saving of figures during test execution.
    ##
    import matplotlib.pyplot as plt
    from unittest.mock import MagicMock

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_fig.gca.return_value = mock_ax
    mock_ax.plot.return_value = None
    mock_ax.axhline.return_value = None
    mock_ax.set_title.return_value = None
    mock_ax.legend.return_value = None

    monkeypatch.setattr(plt, "figure", lambda *a, **k: mock_fig)
    monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
    monkeypatch.setattr(plt, "gca", lambda: mock_ax)
    monkeypatch.setattr(plt, "subplots", lambda *a, **k: (mock_fig, mock_ax))
    monkeypatch.setattr(plt, "tight_layout", lambda *a, **k: None)
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    ##
    # @brief Isolate output by changing to a temporary directory.
    ##
    monkeypatch.chdir(tmp_path)

    ##
    # @brief Dynamically import and execute the main module.
    ##
    import importlib
    main = importlib.import_module("main")

    ##
    # @brief Capture the output and validate summary forecast text.
    ##
    captured = capsys.readouterr()
    assert "Tomorrow's Predicted Prices" in captured.out
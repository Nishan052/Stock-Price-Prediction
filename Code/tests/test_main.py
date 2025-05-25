import os
import pandas as pd
import pytest

def test_main_end_to_end(monkeypatch, tmp_path, capsys):
    # — stub out data loading & modeling —
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    stub_df = pd.DataFrame({"Open": [1, 2], "Close": [1.1, 2.2]}, index=dates)

    import data_handler.data_handler as dh
    import models.arima.arima_model as ar_mod
    import models.lstm.lstm_model as lstm_mod

    monkeypatch.setattr(dh,  "load_nifty50_yfinance", lambda *a, **k: stub_df)
    ar_preds  = {"Open": pd.Series([10,20], index=dates),
                 "Close": pd.Series([11,21], index=dates)}
    ls_preds  = {"Open": pd.Series([12,22], index=dates),
                 "Close": pd.Series([13,23], index=dates)}
    monkeypatch.setattr(ar_mod,   "run_arima", lambda df, columns=None: ar_preds)
    monkeypatch.setattr(lstm_mod,"run_lstm", lambda df, columns=None: ls_preds)

    # — stub out all matplotlib.pyplot calls used in main.py —
    import matplotlib.pyplot as plt
    for fn in ("figure","plot","xlabel","ylabel","title","legend","tight_layout","savefig","show"):
        monkeypatch.setattr(plt, fn, lambda *a, **k: None)

    # switch to tmp_path so files land there
    monkeypatch.chdir(tmp_path)

    # now import & execute main.py
    import importlib
    main = importlib.import_module("main")

    # capture its printout
    captured = capsys.readouterr()
    assert "Tomorrow's Predicted Prices" in captured.out

    # forecast_results.csv was written
    out_csv = tmp_path / "forecast_results.csv"
    assert out_csv.exists()

    df_res = pd.read_csv(out_csv, index_col=0)
    expected_cols = {
        "ARIMA_Open","ARIMA_Close",
        "LSTM_Open","LSTM_Close",
        "Actual_Open","Actual_Close"
    }
    assert set(df_res.columns) == expected_cols
import os
import numpy as np
import pandas as pd
import logging
import joblib
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from config import get_train_end_date, ROLLING_WINDOW_YEARS

# Directory where ARIMA models will be saved
MODEL_DIR = "models/arima"

def run_arima(df, columns=None, retrain=True):
    """
    Runs ARIMA for specified columns using walk-forward forecasting.
    Saves separate model files for each column if retrain=True.

    Args:
        df (DataFrame): Time-indexed data containing target columns (e.g., "Open", "Close").
        columns (list): List of columns to forecast. Defaults to ["Open", "Close"] if None.
        retrain (bool): If True, retrains ARIMA on each rolling window and saves the last model.

    Returns:
        dict: {column_name: pd.Series} containing predicted values for each column.
    """
    if columns is None:
        columns = ["Open", "Close"]

    # Define training end date and split train/test data
    TRAIN_END_DATE = get_train_end_date(df)
    df_train = df.loc[:TRAIN_END_DATE]
    df_test = df.loc[TRAIN_END_DATE:]
    test_index = df_test.index[df_test.index > TRAIN_END_DATE]

    forecasts = {}

    # Rolling window size for training (default is 2 years)
    # Used to simulate real-time forecasting by retraining on sliding windows
    rolling_window = pd.DateOffset(years=ROLLING_WINDOW_YEARS)  # Default is 2 in config.py

    os.makedirs(MODEL_DIR, exist_ok=True)

    for col in columns:
        preds = []
        last_model = None

        for day in test_index:
            # Define training window for this step
            start_train = max(df.index.min(), day - rolling_window)
            end_train = day - pd.Timedelta(days=1)

            # If invalid window, skip prediction
            if start_train >= end_train:
                preds.append(np.nan)
                continue

            y_train = df.loc[start_train:end_train, col].dropna()
            if len(y_train) < 5:
                preds.append(np.nan)
                continue

            try:
                # Hyperparameter: auto_arima is used to find the best (p,d,q) automatically
                # - seasonal=False: we assume daily stock data without seasonal cycles
                # - error_action='ignore': suppress errors when fitting bad models
                # - suppress_warnings=True: cleaner logs
                auto_model = auto_arima(
                    y_train,
                    seasonal=False,
                    error_action='ignore',
                    suppress_warnings=True
                )
                best_order = auto_model.order  # (p,d,q) from auto_arima

                # Fit ARIMA using best_order
                model = ARIMA(y_train, order=best_order).fit()

                # Forecast next time step
                pred = model.forecast(steps=1)
                preds.append(pred.iloc[0])
                last_model = model  # Save for later

            except Exception as e:
                logging.error(f"ARIMA training failed for {col} on {day}: {e}")
                preds.append(np.nan)

        forecasts[col] = pd.Series(data=preds, index=test_index)

        # Save the final trained ARIMA model for this column (optional)
        if retrain and last_model:
            try:
                model_path = os.path.join(MODEL_DIR, f"arima_model_{col.lower()}.pkl")
                joblib.dump(last_model, model_path)
                logging.info(f"Saved ARIMA model for {col} to {model_path}")
            except Exception as e:
                logging.warning(f"Failed to save ARIMA model for {col}: {e}")

    return forecasts
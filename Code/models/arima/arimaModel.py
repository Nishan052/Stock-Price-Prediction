##
# @file arimaModel.py
# @brief Provides ARIMA-based time series forecasting with walk-forward validation.
#
# @details
# This module uses `pmdarima.auto_arima` to determine optimal ARIMA (p, d, q) order,
# trains ARIMA models on rolling 3-year windows, and forecasts target columns like "Open" and "Close".
# The most recent model is saved for reuse.
#
# @date June 2025

##

import os
import numpy as np
import pandas as pd
import logging
import joblib
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from config import getTrainEndDate, rollingWindowYears

from utils.errorHandler import logError

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.messageHandler import MessageHandler

msg = MessageHandler()

##
# @var modelDir
# @brief Directory where trained ARIMA models are stored.
##
modelDir = "models/arima"

##
# @brief Run walk-forward ARIMA forecasting for each target column.
#
# @param df Time-indexed DataFrame containing stock data.
# @param columns List of columns to forecast (e.g., ["Open", "Close"]). Defaults to both if None.
# @param retrain Whether to save the final trained model for reuse. Default: True.
# @return Dictionary with forecasts (Series) per target column.
#
# @details
# - Uses `auto_arima` to find the best ARIMA(p,d,q) order on each training window.
# - Each window covers a 3-year period (defined in `rollingWindowYears`).
# - Forecasts one day ahead for each entry in the test index.
# - Saves the final model to `models/arima/arima_model_<column>.pkl`.
##
def runArima(df, columns=None, retrain=True):
    if columns is None:
        columns = ["Open", "Close"]

    ##
    # @brief Compute the end date for training and create test index
    ##
    trainEndDate = getTrainEndDate(df)
    dfTrain = df.loc[:trainEndDate]
    dfTest = df.loc[trainEndDate:]
    testIndex = dfTest.index[dfTest.index > trainEndDate]

    forecasts = {}

    ##
    # @brief Define the training window length (e.g., 3 years)
    ##
    rollingWindow = pd.DateOffset(years=rollingWindowYears)

    os.makedirs(modelDir, exist_ok=True)

    for col in columns:
        preds = []
        lastModel = None

        for day in testIndex:
            startTrain = max(df.index.min(), day - rollingWindow)
            endTrain = day - pd.Timedelta(days=1)

            ##
            # @brief Skip if the training window is invalid
            ##
            if startTrain >= endTrain:
                preds.append(np.nan)
                continue

            ##
            # @brief Extract training data for target column
            ##
            yTrain = df.loc[startTrain:endTrain, col].dropna()

            ##
            # @brief Minimum requirement: 5 observations
            #
            # @note Fewer data points can cause ARIMA estimation to fail.
            ##
            if len(yTrain) < 5:
                preds.append(np.nan)
                continue

            try:
                ##
                # @brief Use auto_arima to determine best order (p, d, q)
                ##
                autoModel = auto_arima(
                    yTrain,
                    seasonal=False,
                    error_action='ignore',
                    suppress_warnings=True
                )
                bestOrder = autoModel.order

                ##
                # @brief Fit ARIMA model and forecast 1 step ahead
                ##
                model = ARIMA(yTrain, order=bestOrder).fit()
                pred = model.forecast(steps=1)
                preds.append(pred.iloc[0])

                ##
                # @brief Store last successfully trained model for saving
                ##
                lastModel = model

            except Exception as e:
                logError(e, context=f"ARIMA Training - {col} on {day.date()}")
                preds.append(np.nan)

        ##
        # @brief Add predictions to forecast result dictionary
        ##
        forecasts[col] = pd.Series(data=preds, index=testIndex)

        ##
        # @brief Save the last trained model to disk (optional)
        #
        # @details
        # If retraining is enabled and a valid model exists, it is saved as a `.pkl` file.
        ##
        if retrain and lastModel:
            try:
                modelPath = os.path.join(modelDir, f"arimaModel{col.capitalize()}.pkl")
                joblib.dump(lastModel, modelPath)
                logging.info(msg.get("arima_model_saved").format(column=col, path=modelPath))
            except Exception as e:
                logError(e, context=f"ARIMA Save Model - {col}")

    return forecasts
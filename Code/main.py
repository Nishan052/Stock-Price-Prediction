##
# @mainpage NIFTY 50 Stock Price Forecasting App
#
# @section author Author
# Nishan Chandrashekar Poojary  
# Email: nishan.chandrashekar.poojary@stud.hs-emden-leer.com  
#
# Sandesh Nonavinakere Sunil  
# Email: sandesh.nonavinakere.sunil@stud.hs-emden-leer.com  
#
# Created: June 2025
#
# @section overview Project Overview
# This project implements a walk-forward one-step forecasting system for the Indian NIFTY 50 stock index
# using two time series modeling techniques: ARIMA and LSTM.
#
# The system:
# - Collects historical NIFTY 50 data from Yahoo Finance via the yFinance API.
# - Cleans the data, filters for weekdays, and adds a COVID-19 dummy variable for the 2020 pandemic period.
# - Performs predictions using:
#   - **ARIMA**: A statistical model that captures autocorrelation and trends in univariate time series data.
#   - **LSTM**: A deep learning model designed to handle long-term dependencies in multivariate sequences.
# - Uses a rolling window with periodic retraining to maintain adaptability to recent market behavior.
# - Computes error metrics including RMSE and MAPE to evaluate model performance.
# - Displays and saves the forecasted values versus actual prices in graphical form.
#
# The application supports forecasting for both `Open` and `Close` prices, enabling side-by-side comparison
# of model predictions across short-term financial windows.
##

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

## @brief Load modules from internal project structure
from dataHandler.dataHandler import loadNifty50Yfinance
from models.arima.arimaModel import runArima
from models.lstm.lstmModel import runLstm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "utils")))
from utils.messageHandler import MessageHandler
from utils.errorHandler import logError

## @brief Instantiate the message handler for UI messages
msg = MessageHandler()

## @brief Configure logging to capture runtime errors and debug info
logging.basicConfig(
    filename='log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

##
# @section Data Loading
#
# Load historical NIFTY 50 data using the internal loader.
# Errors are logged using the custom handler.
##
try:
    df = loadNifty50Yfinance()
except Exception as e:
    logError(e, context="Data Loading")
    raise

##
# @var targetColumns
# List of features to forecast — limited to 'Open' and 'Close'.
##
targetColumns = ["Open", "Close"]

##
# @brief Run ARIMA and LSTM forecasting on target columns
#
# @details
# Returns dataframes with predictions indexed by date and column name.
##
arimaResults = runArima(df, columns=targetColumns)
lstmResults = runLstm(df, columns=targetColumns)

##
# @section Index Alignment
#
# Find the common forecastable dates between models and filter test data accordingly.
##
arimaIndex = arimaResults["Open"].dropna().index.intersection(arimaResults["Close"].dropna().index)
lstmIndex = lstmResults["Open"].dropna().index.intersection(lstmResults["Close"].dropna().index)
testIndex = arimaIndex.intersection(lstmIndex)

##
# @brief Print the next available day prediction if test data is available.
##
if not testIndex.empty:
    tomorrow = testIndex[0]
    print(msg.get("predicted_prices_header"))
    for col in targetColumns:
        arimaTomorrow = arimaResults[col].loc[tomorrow]
        lstmTomorrow = lstmResults[col].loc[tomorrow]
        print(msg.get("predicted_price_line").format(
            column=col, arima=arimaTomorrow, lstm=lstmTomorrow
        ))
else:
    print(msg.get("no_test_index"))

##
# @section Evaluation
#
# Compute RMSE and MAPE for both ARIMA and LSTM predictions over valid dates.
##
for col in targetColumns:
    actuals = df.loc[testIndex, col]
    arimaPreds = arimaResults[col].reindex(testIndex).dropna()
    lstmPreds = lstmResults[col].reindex(testIndex).dropna()

    validIndex = actuals.index.intersection(arimaPreds.index).intersection(lstmPreds.index)
    actuals = actuals.loc[validIndex]

    if not actuals.empty:
        ##
        # @brief Calculate error metrics
        #
        # @details
        # RMSE: Root Mean Squared Error
        # MAPE: Mean Absolute Percentage Error
        ##
        arimaRmse = np.sqrt(mean_squared_error(actuals, arimaPreds.loc[validIndex]))
        lstmRmse = np.sqrt(mean_squared_error(actuals, lstmPreds.loc[validIndex]))
        arimaMape = mean_absolute_percentage_error(actuals, arimaPreds.loc[validIndex])
        lstmMape = mean_absolute_percentage_error(actuals, lstmPreds.loc[validIndex])

        print(msg.get("forecast_results_header").format(column=col))
        print(msg.get("forecast_metrics").format(model="ARIMA", rmse=arimaRmse, mape=arimaMape))
        print(msg.get("forecast_metrics").format(model="LSTM", rmse=lstmRmse, mape=lstmMape))
    else:
        print(msg.get("no_valid_forecast").format(column=col))

##
# @section Result Aggregation
#
# Combine model predictions and actuals into a single dataframe for plotting and export.
##
results = pd.DataFrame({
    "ARIMA_Open": arimaResults["Open"],
    "ARIMA_Close": arimaResults["Close"],
    "LSTM_Open": lstmResults["Open"],
    "LSTM_Close": lstmResults["Close"],
    "Actual_Open": df["Open"].reindex_like(arimaResults["Open"]),
    "Actual_Close": df["Close"].reindex_like(arimaResults["Close"]),
})

##
# @section Plotting
#
# Generate and save comparison plots for predicted vs. actual values.
##
for col in targetColumns:
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results[f"Actual_{col}"], label="Actual", color="black")
    plt.plot(results.index, results[f"ARIMA_{col}"], label="ARIMA", linestyle="--")
    plt.plot(results.index, results[f"LSTM_{col}"], label="LSTM", linestyle=":")

    plt.xlabel("Date")
    plt.ylabel(f"{col} Price")
    plt.title(f"Actual vs Predicted {col} Prices")
    plt.legend()

    ## @brief Set date formatting for x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    ##
    # @brief Save plot to file
    #
    # @details
    # Plots are saved as PNG files using a naming convention that includes the target column.
    ##
    plt.savefig(f"forecast_plot_{col}.png")
    plt.show()
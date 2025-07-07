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
# This Streamlit application performs one-step-ahead stock price forecasting
# for the NIFTY 50 index using two models:
# - ARIMA (Autoregressive Integrated Moving Average)
# - LSTM (Long Short-Term Memory)
#
# The app provides:
# - Model selection
# - Date input
# - Predictions and comparison to actual prices
# - RMSE and MAPE evaluation metrics
#
# Input data is pulled via yFinance. Models are loaded from disk using
# joblib (for ARIMA) and Keras (for LSTM). GUI is built with Streamlit.
##

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from GUICode.Code.errorHandler import logError
import json

##
# @var baseDir
# Absolute path to the directory containing the current file.
##
baseDir = os.path.dirname(os.path.abspath(__file__))

##
# @var arimaModelPath
# Dictionary storing paths to pretrained ARIMA model files for 'Open' and 'Close' columns.
##
arimaModelPath = {
    "Open": os.path.join(baseDir, "models", "arimaModelOpen.pkl"),
    "Close": os.path.join(baseDir, "models", "arimaModelClose.pkl"),
}

##
# @var lstmModelPath
# Dictionary storing paths to pretrained LSTM model files for 'Open' and 'Close' columns.
##
lstmModelPath = {
    "Open": os.path.join(baseDir, "models", "lstmModelOpen.keras"),
    "Close": os.path.join(baseDir, "models", "lstmModelClose.keras"),
}

##
# @var messagePath
# Path to the JSON file containing user interface messages.
##
messagePath = os.path.join(baseDir, "messages.json")

##
# @var msg
# GUI messages loaded from JSON file.
#
# @var LOG_CTX
# Logging context messages loaded from JSON file.
##
with open(messagePath, "r") as f:
    msg = json.load(f)["gui"]
    LOG_CTX = json.load(open(messagePath))["log_contexts"]

##
# Streamlit UI Configuration and Initialization
##
st.set_page_config(layout="centered")
st.title(msg["title"])
st.markdown("---")

##
# @var typeSelect
# Dropdown selection for the model type: ARIMA or LSTM.
##
typeSelect = st.selectbox("Select Model", ["Select", "ARIMA", "LSTM"])

##
# @var today
# @var tomorrow
# Define current and next date for date input range.
##
today = datetime.today().date()
tomorrow = today + timedelta(days=1)

##
# @var selectedDate
# Date selected by the user for prediction.
##
selectedDate = st.date_input(
    msg["date_input_label"],
    value=today,
    min_value=datetime(2008, 1, 1).date(),
    max_value=tomorrow
)

##
# Block user from selecting weekends (non-trading days)
##
if selectedDate.weekday() in [5, 6]:
    st.error(msg["weekend_error"])
    st.stop()

##
# Execute forecasting logic when prediction button is clicked.
##
if st.button(msg["predict_button"]):
    if typeSelect == "Select":
        st.warning(msg["select_model_warning"])
    else:
        try:
            ##
            # @var predictionDay
            # Parsed datetime of the selected date.
            #
            # @var cutoffDay
            # One day before prediction for training end.
            ##
            predictionDay = pd.to_datetime(selectedDate)
            cutoffDay = predictionDay - timedelta(days=1)

                        ##
            # @brief Download historical NIFTY 50 data using yFinance
            #
            # @details
            # Data is fetched from January 1, 2008 up to (cutoffDay + 1) to include
            # the latest available data for prediction. The 2008 start date is chosen
            # to ensure a sufficient history while reducing memory and bandwidth load.
            # Only essential columns are retained: Open, High, Low, Close, Volume.
            ##
            df = yf.download("^NSEI", start="2008-01-01", end=(cutoffDay + timedelta(days=1)).strftime("%Y-%m-%d"))
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()

            ##
            # @brief Validate minimum required data length
            #
            # @details
            # A minimum of 61 rows (days) is required for a 60-day lookback LSTM input
            # and for training ARIMA with enough historical context.
            ##
            if len(df) < 61:
                st.error(msg["insufficient_data"])
                st.stop()

            ##
            # @var predictions
            # Stores predicted values for 'Open' and 'Close' using the selected model.
            ##
            predictions = {}

            ##
            # @brief Generate prediction using ARIMA model
            #
            # @details
            # For each target column, the pretrained model is loaded from disk.
            # Its `order` configuration is reused to fit a new ARIMA model on the
            # freshly downloaded data to preserve forecasting logic.
            # One-step forecast is extracted using `.forecast(steps=1)`.
            ##
            if typeSelect == "ARIMA":
                for col in ["Open", "Close"]:
                    modelPath = arimaModelPath[col]          ## Path to pretrained ARIMA model for current column
                    model = joblib.load(modelPath)              ## Load pretrained ARIMA model object
                    order = model.model.order                    ## Extract (p,d,q) order used during training

                    ##
                    # Refit model on the most recent data using extracted order
                    ##
                    fittedModel = ARIMA(df[col], order=order).fit()
                    forecast = fittedModel.forecast(steps=1)   ## One-day-ahead forecast
                    predictions[col] = float(forecast.iloc[0])  ## Store result in dictionary

            ##
            # @brief Generate prediction using LSTM model
            #
            # @details
            # For each target column, a 60-day sequence is scaled and reshaped to form
            # the input tensor expected by the LSTM model. The predicted scaled output is
            # inverse-transformed using the fitted MinMaxScaler to get the original price.
            ##
            elif typeSelect == "LSTM":
                ##
                # @var lookback
                # Number of past days used as input to the LSTM model.
                #
                # @note
                # 60 is a conventional value providing ~3 months of trading history.
                # Enough for learning short-term trends without overfitting.
                ##
                lookback = 60

                ##
                # @brief Normalize the 'Open' and 'Close' features using MinMaxScaler
                #
                # @details
                # Scaling ensures LSTM receives data in a consistent numeric range [0,1].
                # Inverse transform is later used to retrieve real-world predictions.
                ##
                scaler = MinMaxScaler()
                scaledData = scaler.fit_transform(df[["Open", "Close"]])
                scaledDf = pd.DataFrame(scaledData, columns=["Open", "Close"], index=df.index)

                ##
                # @brief Prepare the last 60 days as LSTM input
                #
                # @details
                # LSTM expects 3D input shape: (samples, timesteps, features)
                # So we wrap the 60-day, 2-feature input into a 3D NumPy array.
                ##
                sequence = scaledDf[["Open", "Close"]].values[-lookback:]
                xInput = np.array([sequence])  ## Shape: (1, 60, 2)

                ##
                # @brief Load pretrained model and predict for both 'Open' and 'Close'
                #
                # @details
                # Because LSTM models were trained independently for each output,
                # we loop over the target columns and make individual predictions.
                ##
                for col in ["Open", "Close"]:
                    modelPath = lstmModelPath[col]                ## Load model specific to 'Open' or 'Close'
                    model = load_model(modelPath, compile=False)     ## Load pre-trained model without compiling
                    yScaled = model.predict(xInput, verbose=0)      ## Run prediction on input sequence

                    ##
                    # @brief Prepare inverse transformation for single output value
                    #
                    # @details
                    # We create a dummy 2D array of shape (1, 2) filled with zeros,
                    # then insert the predicted scaled value at the correct index.
                    # This is required because inverse_transform expects all feature dimensions.
                    ##
                    temp = np.zeros((1, 2))
                    idx = ["Open", "Close"].index(col)
                    temp[0, idx] = yScaled[0][0]

                    ##
                    # @brief Inverse scale to get original price prediction
                    ##
                    yPred = scaler.inverse_transform(temp)[0][idx]
                    predictions[col] = float(yPred)

            ##
            # @brief Download full historical data for plotting and actual value lookup
            #
            # This data includes all 'Open' and 'Close' values from 2000 onwards
            # and is used to compare predicted values with actuals and for evaluation metrics.
            ##
            dfFull = yf.download("^NSEI", start="2008-01-01", end=(cutoffDay + timedelta(days=1)).strftime("%Y-%m-%d"))
            dfFull = dfFull[["Open", "Close"]].dropna().sort_index()

            ##
            # @brief Display prediction results and actual values with plots
            ##
            for col in ["Open", "Close"]:
                st.subheader(msg["prediction_header"].format(col=col, date=selectedDate))
                st.success(msg["prediction_success"].format(model=typeSelect, col=col, value=predictions[col]))

                ##
                # @brief Attempt to retrieve actual value for the selected date
                ##
                try:
                    actualVal = dfFull.loc[dfFull.index == pd.to_datetime(selectedDate), col]
                    if not actualVal.empty:
                        actual = float(actualVal.values[0])
                        st.info(msg["actual_success"].format(col=col, date=selectedDate, value=actual))
                    else:
                        st.error(msg["actual_data_missing"].format(col=col, date=selectedDate))
                        actual = None
                except Exception as e:
                    logError(LOG_CTX["fetching_actuals"].format(col=col), e)
                    st.warning(msg["actual_fetch_error"].format(col=col))

                ##
                # @brief Plot predicted and actual values on top of recent trend
                ##
                if actual is not None:
                    fig, ax = plt.subplots()
                    ax.plot(dfFull[col].loc[:cutoffDay].tail(30), label="Recent Prices")

                    ##
                    # @note The line y=predictions[col] shows the model's predicted value
                    #       for the selected future date. The dashed style distinguishes it.
                    ##
                    ax.axhline(y=predictions[col], color="red", linestyle="--", label="Predicted")

                    ##
                    # @note The dotted line y=actual shows the actual stock price (if available)
                    ##
                    ax.axhline(y=actual, color="gray", linestyle=":", label="Actual")

                    ax.set_title(f"{col} - Predicted vs Actual")
                    ax.legend()
                    fig.autofmt_xdate()
                    st.pyplot(fig)

            ##
            # @brief Prepare RMSE and MAPE summary table
            #
            # @var lastN
            # Number of days used for computing rolling prediction error. Set to 30
            # to represent a typical business month of trading days.
            ##
            summaryRows = []
            lastN = 30

            for col in ["Open", "Close"]:
                actualSeries = dfFull[col].loc[:cutoffDay].tail(lastN)
                predictedSeries = [predictions[col]] * len(actualSeries)

                ##
                # @brief Calculate RMSE (Root Mean Squared Error)
                #
                # Measures average magnitude of the error, penalizing larger deviations.
                ##
                rmse = np.sqrt(np.mean((np.array(actualSeries) - np.array(predictedSeries)) ** 2))

                ##
                # @brief Calculate MAPE (Mean Absolute Percentage Error)
                #
                # Measures the average percentage error relative to actual values.
                # A small epsilon (1e-10) is added to the denominator to avoid divide-by-zero.
                ##
                mape = np.mean(np.abs((np.array(actualSeries) - np.array(predictedSeries)) /
                                      (np.array(actualSeries) + 1e-10))) * 100

                pred = predictions[col]
                lowerRmse = pred - rmse
                upperRmse = pred + rmse
                lowerMape = pred * (1 - mape / 100)
                upperMape = pred * (1 + mape / 100)

                ##
                # @brief Retrieve actual value again for table display
                ##
                actualVal = dfFull.loc[dfFull.index == pd.to_datetime(selectedDate), col]
                actual = float(actualVal.values[0]) if not actualVal.empty else None

                summaryRows.append({
                    "Type": col,
                    "RMSE": f"{rmse:.2f}",
                    "RMSE Range": f"{lowerRmse:.2f} - {upperRmse:.2f}",
                    "MAPE": f"{mape:.2f}%",
                    "MAPE Range": f"{lowerMape:.2f} - {upperMape:.2f}",
                    "Predicted": f"{pred:.2f}",
                    "Actual": f"{actual:.2f}" if actual is not None else "N/A"
                })

            ##
            # @brief Display summary performance table in the Streamlit app
            ##
            summaryDf = pd.DataFrame(summaryRows)
            st.markdown(msg["summary_table_title"])
            st.table(summaryDf)

            ##
            # @brief Explain RMSE and MAPE metrics below the table
            ##
            st.markdown(msg["rmseMape_explanation"])

        ##
        # @brief Handle exceptions during the entire prediction pipeline
        #
        # This ensures user-friendly error messages and logs detailed
        # exception traces for debugging.
        ##
        except Exception as e:
            logError(LOG_CTX["prediction_block"], e)
            st.error(msg["prediction_failed"].format(error=str(e)))
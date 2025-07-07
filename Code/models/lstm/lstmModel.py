##
# @file lstmModel.py
# @brief Contains the LSTM model logic for forecasting NIFTY 50 stock prices.
#
# @details
# This module handles:
# - Rolling window training of LSTM models
# - Walk-forward validation
# - Scaling and inverse transformation
# - Model saving and logging
#
# Models are trained for each column independently ('Open', 'Close') and saved as `.keras`.
# Scalers are saved using joblib for consistent inverse transformation.
#
# @date June 2025

##

import os
import numpy as np
import pandas as pd
import logging
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from config import getTrainEndDate, lookback, retrainInterval, rollingWindowYears
from utils.errorHandler import logError

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.messageHandler import MessageHandler

msg = MessageHandler()

##
# @brief Builds and trains an LSTM model on scaled training data.
#
# @param dfTrainScaled Scaled training dataframe with all features.
# @param targetColumn The name of the column to predict.
# @param lookback Number of past days to use for each sequence (default = lookback).
# @param epochs Number of training epochs (default = 5).
# @param batchSize Mini-batch size for gradient descent (default = 32).
# @return Trained Keras Sequential model or None if training data is insufficient.
#
# @details
# LSTM(64) + Dropout(0.2) + LSTM(32) + Dropout(0.2) + Dense(1)
# is a common architecture to avoid overfitting and capture temporal patterns.
##
def buildAndTrainLstm(dfTrainScaled, targetColumn, lookback=lookback, epochs=5, batchSize=32):
    ##
    # @brief Internal function to create supervised learning sequences.
    #
    # @param dataframe Scaled dataframe of features.
    # @param lookback Number of past time steps to use.
    # @return Tuple (X, y) of training sequences.
    ##
    def createSequences(dataframe, lookback):
        X, y = [], []
        for i in range(len(dataframe) - lookback):
            seq_x = dataframe.iloc[i:i+lookback].values
            seq_y = dataframe.iloc[i+lookback][targetColumn]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    xTrain, yTrain = createSequences(dfTrainScaled, lookback)
    if len(xTrain) == 0:
        return None

    numFeatures = xTrain.shape[2]

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, numFeatures)))  # First LSTM layer
    model.add(Dropout(0.2))  # Dropout to prevent overfitting
    model.add(LSTM(32))      # Second LSTM layer
    model.add(Dropout(0.2))  # Dropout again
    model.add(Dense(1))      # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xTrain, yTrain, epochs=epochs, batch_size=batchSize, verbose=0, shuffle=False)

    return model
##
# @brief Executes walk-forward LSTM training and forecasting for given time series columns.
#
# @param df DataFrame containing historical stock prices with datetime index.
# @param columns List of target columns to forecast. Default is ["Open", "Close"].
# @return Dictionary of predicted Series (one per column) indexed by date.
#
# @details
# - Trains a separate model for each column using a 3-year rolling window.
# - Retrains every `retrainInterval` days to stay adaptive.
# - Forecasts one step ahead, scales input, and applies inverse transform.
# - Models and scalers are saved for reuse.
##
def runLstm(df, columns=["Open", "Close"]):
    ##
    # @var trainEndDate
    # Date to split training and test data based on config.
    ##
    trainEndDate = getTrainEndDate(df)
    dfTrain = df.loc[:trainEndDate]
    dfTest = df.loc[trainEndDate:]

    ##
    # @var testIndex
    # Dates used for evaluation; all points after training end date.
    ##
    testIndex = dfTest.index[dfTest.index > trainEndDate]

    ##
    # @var rollingWindow
    # Length of rolling training window (e.g., 3 years) defined in config.
    ##
    rollingWindow = pd.DateOffset(years=rollingWindowYears)

    ##
    # @brief Ensure model directory exists before saving.
    ##
    os.makedirs("models/lstm", exist_ok=True)

    predictions = {}

    for targetColumn in columns:
        preds = []
        lstmModelCurrent = None
        lstmScalerCurrent = None
        lstmCounter = 0

        for day in testIndex:
            startTrain = max(df.index.min(), day - rollingWindow)
            endTrain = day - pd.Timedelta(days=1)

            ##
            # Skip if the training window is invalid (not enough history).
            ##
            if startTrain >= endTrain:
                preds.append(np.nan)
                continue

            dfWindow = df.loc[startTrain:endTrain, columns].dropna()

            ##
            # @brief Require at least lookback + 1 rows for valid supervised learning.
            ##
            if len(dfWindow) < lookback + 1:
                logging.debug(msg.get("lstm_training_skipped_insufficient_data").format(date=day.date()))
                preds.append(np.nan)
                continue

            lstmCounter += 1

            ##
            # @brief Retrain model every retrainInterval steps or on first iteration.
            ##
            if (lstmCounter % retrainInterval == 0) or (lstmModelCurrent is None):
                try:
                    ##
                    # @brief Fit a MinMaxScaler and scale training data
                    ##
                    scaler = MinMaxScaler()
                    scaled_train = pd.DataFrame(
                        scaler.fit_transform(dfWindow),
                        index=dfWindow.index,
                        columns=columns
                    )

                    ##
                    # @brief Train LSTM model using scaled data
                    ##
                    lstmModel = buildAndTrainLstm(scaled_train, targetColumn)

                    ##
                    # @brief Store current model and scaler for reuse
                    ##
                    lstmModelCurrent = lstmModel
                    lstmScalerCurrent = scaler

                    ##
                    # @brief Save model and scaler to disk for inspection or reuse
                    ##
                    modelPath = os.path.join("models", "lstm", f"lstmModel{targetColumn.capitalize()}.keras")
                    scaler_path = os.path.join("models", "lstm", f"scaler{targetColumn.capitalize()}.pkl")
                    lstmModel.save(modelPath)
                    joblib.dump(scaler, scaler_path)
                    logging.info(msg.get("lstm_model_saved").format(column=targetColumn))

                except Exception as e:
                    logError(e, context=f"LSTM Training - {targetColumn} on {day.date()}")
                    preds.append(np.nan)
                    continue
            else:
                lstmModel = lstmModelCurrent
                scaler = lstmScalerCurrent

            if lstmModel is None or scaler is None:
                preds.append(np.nan)
                continue

            try:
                ##
                # @brief Collect the last `lookback` days prior to prediction day
                ##
                pastIdx = df.index[df.index < day][-lookback:]

                ##
                # @brief Skip if we don’t have enough days for a complete input sequence
                ##
                if len(pastIdx) < lookback:
                    preds.append(np.nan)
                    continue

                dfTestWindow = df.loc[pastIdx.union([day]), columns].dropna()
                if dfTestWindow.shape[0] < lookback + 1:
                    preds.append(np.nan)
                    continue

                ##
                # @brief Scale the test input window using the same scaler as training
                ##
                scaledWindow = pd.DataFrame(
                    scaler.transform(dfTestWindow),
                    index=dfTestWindow.index,
                    columns=columns
                )

                ##
                # @brief Prepare input for LSTM: shape (1, lookback, features)
                ##
                xInput = np.array([scaledWindow.iloc[0:lookback].values])

                ##
                # @brief Predict scaled output for the next step
                ##
                yPredScaled = lstmModel.predict(xInput, verbose=0)

                ##
                # @brief Prepare dummy input for inverse_transform
                #
                # Only the target column is filled with the predicted value;
                # other values remain zero as placeholders.
                ##
                tempInput = np.zeros((1, len(columns)))
                col_idx = columns.index(targetColumn)
                tempInput[0, col_idx] = yPredScaled.flatten()[0]

                ##
                # @brief Convert prediction back to original scale
                ##
                inverseResult = scaler.inverse_transform(tempInput)
                preds.append(inverseResult[0, col_idx])

            except Exception as e:
                logError(e, context=f"LSTM Prediction - {targetColumn} on {day.date()}")
                preds.append(np.nan)

        ##
        # @brief Store predictions for this target column as a Series
        ##
        predictions[targetColumn] = pd.Series(data=preds, index=testIndex)

    return predictions
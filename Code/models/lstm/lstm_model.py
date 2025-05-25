import os
import numpy as np
import pandas as pd
import logging
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from config import get_train_end_date, LOOKBACK, RETRAIN_INTERVAL, ROLLING_WINDOW_YEARS

def build_and_train_lstm(df_train_scaled, target_column, lookback=LOOKBACK, epochs=5, batch_size=32):
    """
    Builds and trains a two-layer LSTM model for forecasting a single column.

    Args:
        df_train_scaled (pd.DataFrame): Scaled training data (e.g., using MinMaxScaler).
        target_column (str): The column to predict ("Open" or "Close").
        lookback (int): How many past days are used as input. Default is 60, which balances historical context and model complexity.
        epochs (int): Number of training epochs. Default is 5 to keep training fast and avoid overfitting for small windows.
        batch_size (int): Number of samples per training batch. Default 32 is a common choice balancing memory and stability.

    Returns:
        A trained LSTM Keras model.
    """
    def create_sequences(dataframe, lookback):
        X, y = [], []
        for i in range(len(dataframe) - lookback):
            seq_x = dataframe.iloc[i:i+lookback].values
            seq_y = dataframe.iloc[i+lookback][target_column]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(df_train_scaled, lookback)
    if len(X_train) == 0:
        return None

    num_features = X_train.shape[2]

    # Model architecture: two LSTM layers and two dropout layers
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, num_features)))
    # 64 units allow the network to learn moderately complex temporal dependencies
    model.add(Dropout(0.2))  # 20% dropout to reduce overfitting
    model.add(LSTM(32))      # 32 units further downsample features before prediction
    model.add(Dropout(0.2))
    model.add(Dense(1))      # Output single prediction value
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

    return model

def run_lstm(df, columns=["Open", "Close"]):
    """
    Runs LSTM walk-forward forecasting for each specified column.

    Args:
        df (pd.DataFrame): Full time series dataset with a datetime index.
        columns (list): Which columns to predict. Defaults to ["Open", "Close"].

    Returns:
        dict: Mapping from column name to Series of walk-forward predictions.
    """
    TRAIN_END_DATE = get_train_end_date(df)
    df_train = df.loc[:TRAIN_END_DATE]
    df_test = df.loc[TRAIN_END_DATE:]
    test_index = df_test.index[df_test.index > TRAIN_END_DATE]

    # Use last 2 years (default) for each rolling training window
    rolling_window = pd.DateOffset(years=ROLLING_WINDOW_YEARS)
    os.makedirs("models/lstm", exist_ok=True)

    predictions = {}

    for target_column in columns:
        preds = []
        lstm_model_current = None
        lstm_scaler_current = None
        lstm_counter = 0

        for day in test_index:
            start_train = max(df.index.min(), day - rolling_window)
            end_train = day - pd.Timedelta(days=1)

            if start_train >= end_train:
                preds.append(np.nan)
                continue

            df_window = df.loc[start_train:end_train, columns].dropna()
            if len(df_window) < LOOKBACK + 1:
                # Need at least lookback+1 rows to train and predict
                logging.debug(f"[{day.date()}] Skipped due to insufficient window for training.")
                preds.append(np.nan)
                continue

            lstm_counter += 1
            # Retrain model every RETRAIN_INTERVAL steps, or if model is not initialized
            if (lstm_counter % RETRAIN_INTERVAL == 0) or (lstm_model_current is None):
                try:
                    scaler = MinMaxScaler()
                    scaled_train = pd.DataFrame(
                        scaler.fit_transform(df_window),
                        index=df_window.index,
                        columns=columns
                    )
                    lstm_model = build_and_train_lstm(scaled_train, target_column)
                    lstm_model_current = lstm_model
                    lstm_scaler_current = scaler

                    # Save model and scaler
                    model_path = os.path.join("models", "lstm", f"lstm_model_{target_column.lower()}.keras")
                    scaler_path = os.path.join("models", "lstm", f"scaler_{target_column.lower()}.pkl")
                    lstm_model.save(model_path)
                    joblib.dump(scaler, scaler_path)
                    logging.info(f"Saved LSTM model and scaler for '{target_column}'")

                except Exception as e:
                    logging.error(f"LSTM Train Error for {target_column} on {day}: {e}")
                    preds.append(np.nan)
                    continue
            else:
                lstm_model = lstm_model_current
                scaler = lstm_scaler_current

            if lstm_model is None or scaler is None:
                preds.append(np.nan)
                continue

            try:
                # Use past 60 days for prediction
                past_idx = df.index[df.index < day][-LOOKBACK:]
                if len(past_idx) < LOOKBACK:
                    preds.append(np.nan)
                    continue

                df_test_window = df.loc[past_idx.union([day]), columns].dropna()
                if df_test_window.shape[0] < LOOKBACK + 1:
                    preds.append(np.nan)
                    continue

                scaled_window = pd.DataFrame(
                    scaler.transform(df_test_window),
                    index=df_test_window.index,
                    columns=columns
                )

                X_input = np.array([scaled_window.iloc[0:LOOKBACK].values])
                y_pred_scaled = lstm_model.predict(X_input, verbose=0)

                # Inverse scaling for single predicted column
                temp_input = np.zeros((1, len(columns)))
                col_idx = columns.index(target_column)
                temp_input[0, col_idx] = y_pred_scaled.flatten()[0]
                inverse_result = scaler.inverse_transform(temp_input)
                preds.append(inverse_result[0, col_idx])
            except Exception as e:
                logging.error(f"LSTM Predict Error for {target_column} on {day}: {e}")
                preds.append(np.nan)

        predictions[target_column] = pd.Series(data=preds, index=test_index)

    return predictions
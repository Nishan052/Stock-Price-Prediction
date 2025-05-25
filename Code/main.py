import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Import data loading and modeling modules
from data_handler.data_handler import load_nifty50_yfinance
from models.arima.arima_model import run_arima
from models.lstm.lstm_model import run_lstm
from config import get_train_end_date
import matplotlib.dates as mdates

# --- Logging Setup ---
# Configure logging to save debug and error messages to log.txt
logging.basicConfig(filename='log.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Loading ---
# Try to load and clean NIFTY 50 data using custom data handler.
try:
    df = load_nifty50_yfinance()
except Exception as e:
    logging.error(f"Data loading error: {e}")
    raise

# --- Model Setup ---
# Specify which columns will be predicted.
target_columns = ["Open", "Close"]

# Run ARIMA and LSTM forecasting functions; outputs are dicts of DataFrames keyed by column.
arima_results = run_arima(df, columns=target_columns)
lstm_results = run_lstm(df, columns=target_columns)

# --- Prediction Index Preparation ---
# Find the set of dates for which all models produced a prediction (and both columns are available).
arima_index = arima_results["Open"].dropna().index.intersection(arima_results["Close"].dropna().index)
lstm_index = lstm_results["Open"].dropna().index.intersection(lstm_results["Close"].dropna().index)
test_index = arima_index.intersection(lstm_index)

# --- Next-Day Price Prediction Display ---
# Print ARIMA and LSTM predictions for the first available date in the test set.
if not test_index.empty:
    tomorrow = test_index[0]
    print("====== Tomorrow's Predicted Prices ======")
    for col in target_columns:
        arima_tomorrow = arima_results[col].loc[tomorrow]
        lstm_tomorrow = lstm_results[col].loc[tomorrow]
        print(f"{col} - ARIMA: {arima_tomorrow:.4f}, LSTM: {lstm_tomorrow:.4f}")
else:
    print("Test index is empty; unable to extract tomorrow's prediction.")

# --- Model Evaluation ---
# Evaluate model predictions (RMSE, MAPE) against actual values for each column.
for col in target_columns:
    actuals = df.loc[test_index, col]  # True values for test set
    arima_preds = arima_results[col].reindex(test_index).dropna()
    lstm_preds = lstm_results[col].reindex(test_index).dropna()

    # Only use dates available in all three: actuals, ARIMA, and LSTM
    valid_index = actuals.index.intersection(arima_preds.index).intersection(lstm_preds.index)
    actuals = actuals.loc[valid_index]

    if not actuals.empty:
        # Calculate error metrics for each model on valid dates
        arima_rmse = np.sqrt(mean_squared_error(actuals, arima_preds.loc[valid_index]))
        lstm_rmse = np.sqrt(mean_squared_error(actuals, lstm_preds.loc[valid_index]))
        arima_mape = mean_absolute_percentage_error(actuals, arima_preds.loc[valid_index])
        lstm_mape = mean_absolute_percentage_error(actuals, lstm_preds.loc[valid_index])

        print(f"\n====== {col} Forecast Results ======")
        print(f"ARIMA -> RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2f}")
        print(f"LSTM  -> RMSE: {lstm_rmse:.2f}, MAPE: {lstm_mape:.2f}")
    else:
        print(f"No valid actuals or predictions for {col}.")

# --- Save Forecast Results to CSV ---
# Combine all predictions and actuals into one DataFrame for further analysis or sharing.
results = pd.DataFrame({
    "ARIMA_Open": arima_results["Open"],
    "ARIMA_Close": arima_results["Close"],
    "LSTM_Open": lstm_results["Open"],
    "LSTM_Close": lstm_results["Close"],
    "Actual_Open": df["Open"].reindex_like(arima_results["Open"]),
    "Actual_Close": df["Close"].reindex_like(arima_results["Close"]),
})
results.to_csv("forecast_results.csv", index=True)

# --- Visualization ---
# Plot and save graphs of Actual vs Predicted prices for each column.

for col in target_columns:
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results[f"Actual_{col}"], label="Actual", color="black")
    plt.plot(results.index, results[f"ARIMA_{col}"], label="ARIMA", linestyle="--")
    plt.plot(results.index, results[f"LSTM_{col}"], label="LSTM", linestyle=":")

    plt.xlabel("Date")
    plt.ylabel(f"{col} Price")
    plt.title(f"Actual vs Predicted {col} Prices")
    plt.legend()

    # Format date ticks for better readability
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)  # Rotate x-axis date labels

    plt.tight_layout()
    plt.savefig(f"forecast_plot_{col}.png")  # Save plot as PNG file
    plt.show()  # Display plot in supported environments
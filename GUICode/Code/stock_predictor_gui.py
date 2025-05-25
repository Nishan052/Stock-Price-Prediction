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
import logging


# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARIMA_MODEL_PATHS = {
    "Open": os.path.join(BASE_DIR, "models", "arima_model_open.pkl"),
    "Close": os.path.join(BASE_DIR, "models", "arima_model_close.pkl"),
}
LSTM_MODEL_PATHS = {
    "Open": os.path.join(BASE_DIR, "models", "lstm_model_open.keras"),
    "Close": os.path.join(BASE_DIR, "models", "lstm_model_close.keras"),
}
LOG_PATH = os.path.join(BASE_DIR, "streamlit_app.log")

# Logging configuration
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Streamlit UI setup
st.set_page_config(layout="centered")
st.title("\U0001F4C8 NIFTY 50 Stock Price Predictor")
st.markdown("---")

# Model selector
type_select = st.selectbox("Select Model", ["Select", "ARIMA", "LSTM"])

# Date input
today = datetime.today().date()
tomorrow = today + timedelta(days=1)

selected_date = st.date_input(
    "Select prediction date (based on historical data)",
    value=today,
    min_value=datetime(2007, 1, 1).date(),   # or your data start date
    max_value=tomorrow
)
# Check for weekend selection
if selected_date.weekday() in [5, 6]:  # 5 = Saturday, 6 = Sunday
    st.error("Weekend selected. Please choose a weekday (Monday to Friday).")
    st.stop()

# Prediction trigger
if st.button("Predict Open & Close for Selected Date"):
    if type_select == "Select":
        st.warning("Please select a model.")
    else:
        try:
            prediction_day = pd.to_datetime(selected_date)
            cutoff_day = prediction_day - timedelta(days=1)

            # Load historical data up to cutoff_day
            df = yf.download("^NSEI", start="2000-01-01", end=(cutoff_day + timedelta(days=1)).strftime("%Y-%m-%d"))
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()

            if len(df) < 61:
                st.error("Not enough historical data available for prediction.")
                logging.warning(f"Insufficient data before {selected_date}")
                st.stop()

            predictions = {}

            if type_select == "ARIMA":
                for col in ["Open", "Close"]:
                    model_path = ARIMA_MODEL_PATHS[col]
                    model = joblib.load(model_path)
                    order = model.model.order
                    fitted_model = ARIMA(df[col], order=order).fit()
                    forecast = fitted_model.forecast(steps=1)
                    predictions[col] = float(forecast.iloc[0])

            elif type_select == "LSTM":
                LOOKBACK = 60
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[["Open", "Close"]])
                scaled_df = pd.DataFrame(scaled_data, columns=["Open", "Close"], index=df.index)

                sequence = scaled_df[["Open", "Close"]].values[-LOOKBACK:]
                X_input = np.array([sequence])

                for col in ["Open", "Close"]:
                    model_path = LSTM_MODEL_PATHS[col]
                    model = load_model(model_path, compile=False)
                    y_scaled = model.predict(X_input, verbose=0)

                    temp = np.zeros((1, 2))
                    idx = ["Open", "Close"].index(col)
                    temp[0, idx] = y_scaled[0][0]
                    y_pred = scaler.inverse_transform(temp)[0][idx]
                    predictions[col] = float(y_pred)

            # Load actuals (including prediction day)
            df_full = yf.download("^NSEI", start="2000-01-01")
            df_full = df_full[["Open", "Close"]].dropna().sort_index()

            for col in ["Open", "Close"]:
                st.subheader(f"{col} Price Prediction for {selected_date}")
                st.success(f"[{type_select}] Predicted {col}: {predictions[col]:.2f}")

                try:
                    actual_val = df_full.loc[df_full.index == pd.to_datetime(selected_date), col]
                    if not actual_val.empty:
                        actual = float(actual_val.values[0])
                        st.info(f"Actual {col} on {selected_date}: {actual:.2f}")
                    else:
                        st.error(f"Actual {col} data for {selected_date} is not available.")
                        actual = None
                except Exception as e:
                    logging.error(f"Failed to retrieve actuals for {col}: {e}")
                    st.warning(f"Could not fetch actual {col} price.")

                # Plot
                # Only plot if actual value is available
                if actual is not None:
                    fig, ax = plt.subplots()
                    ax.plot(df_full[col].loc[:cutoff_day].tail(30), label="Recent Prices")
                    ax.axhline(y=predictions[col], color="red", linestyle="--", label="Predicted")
                    ax.axhline(y=actual, color="gray", linestyle=":", label="Actual")
                    ax.set_title(f"{col} - Predicted vs Actual")
                    ax.legend()
                    fig.autofmt_xdate()
                    st.pyplot(fig)
                # ... (existing code for plotting)

                # Build summary table with RMSE and MAPE-based prediction ranges
                summary_rows = []
                last_n = 30

                for col in ["Open", "Close"]:
                    # Calculate RMSE and MAPE using last 30 days
                    actual_series = df_full[col].loc[:cutoff_day].tail(last_n)
                    predicted_series = [predictions[col]] * len(actual_series)
                    rmse = np.sqrt(np.mean((np.array(actual_series) - np.array(predicted_series)) ** 2))
                    mape = np.mean(np.abs((np.array(actual_series) - np.array(predicted_series)) / (np.array(actual_series) + 1e-10))) * 100

                    pred = predictions[col]
                    # RMSE-based range
                    lower_rmse = pred - rmse
                    upper_rmse = pred + rmse
                    # MAPE-based range
                    lower_mape = pred * (1 - mape / 100)
                    upper_mape = pred * (1 + mape / 100)

                    # Actual value for the selected date
                    actual_val = df_full.loc[df_full.index == pd.to_datetime(selected_date), col]
                    actual = float(actual_val.values[0]) if not actual_val.empty else None

                    summary_rows.append({
                        "Type": col,
                        "RMSE Range": f"{lower_rmse:.2f} - {upper_rmse:.2f}",
                        "MAPE Range": f"{lower_mape:.2f} - {upper_mape:.2f}",
                        "Predicted": f"{pred:.2f}",
                        "Actual": f"{actual:.2f}" if actual is not None else "N/A"
                    })

            summary_df = pd.DataFrame(summary_rows)
            st.markdown("### Prediction Summary Table")
            st.table(summary_df)

            # Add explanation (once)
            st.markdown("""
            ### ℹ️ RMSE and MAPE Range Explanation

            - **RMSE Range**: This shows the expected absolute fluctuation around the predicted price. It reflects typical deviations in raw price units.

            - **MAPE Range**: This shows the percentage-based uncertainty range around the predicted value. It adjusts for scale by showing likely percent-based error.

            These ranges help interpret the model's accuracy, not provide financial advice.
            """)
        except Exception as e:
            logging.exception("Prediction failed:")
            st.error(f"Prediction failed: {str(e)}")
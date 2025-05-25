# NIFTY 50 Forecasting Project

This project implements a walk‑forward one‑step forecasting framework for the NIFTY 50 stock index using three different modeling approaches:

* **ARIMA:** A univariate time series model.
* **SARIMAX:** An extension of ARIMA that incorporates exogenous variables.
* **LSTM:** A deep learning model designed to capture non‑linear relationships.

The project covered data loading, cleaning, and exploratory data analysis (EDA), as well as model tuning (using `auto_arima` from pmdarima for ARIMA order selection), walk‑forward forecasting, and performance evaluation with visualizations.

---

## Table of Contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Environment Setup](#environment-setup)

  * [Creating a Virtual Environment](#creating-a-virtual-environment)
  * [Installing Required Packages](#installing-required-packages)
* [Usage](#usage)
* [Testing](#testing)
* [Continuous Integration](#continuous-integration)


---

## Overview

This project forecasted the daily closing price of the NIFTY 50 index using historical data from 2000 to 2024. The dataset included:

* Daily **Open, High, Low, Close** prices.
* Fundamental indicators such as **P/E**, **P/B**, and **Div Yield %**.
* A **COVID\_dummy** to capture the market impact during COVID.

A walk‑forward one‑step forecasting approach was used, where models were updated daily (or at set intervals) using the most recent data. For ARIMA and SARIMAX, `auto_arima` from pmdarima selected the best (p, d, q) order on a rolling training window. The LSTM model was built using TensorFlow/Keras and was retrained periodically for efficiency.

---

## Prerequisites

Before running the project, ensure you had:

* **Python 3.10 or 3.11** (recommended for TensorFlow compatibility)
* Basic familiarity with Python, pandas, NumPy, and time‑series analysis
* A terminal (or command prompt) to run commands

---

## Environment Setup

### Creating a Virtual Environment

1. Open Terminal and navigate to your project directory:

   ```bash
   cd /path/to/your/project
   ```
2. Create a virtual environment (using Python 3.10 explicitly):

   ```bash
   python3.10 -m venv myenv310
   ```
3. Activate the virtual environment:

   * On macOS/Linux:

     ```bash
     source myenv310/bin/activate
     ```
   * On Windows:

     ```powershell
     myenv310\Scripts\activate
     ```

### Installing Required Packages

1. Install runtime dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Install developer dependencies (tests, notebooks, Streamlit):

   ```bash
   pip install -r dev-requirements.txt
   ```
3. Verify the setup:

   ```bash
   python --version
   pip list
   ```

---

## Usage

To run the full forecasting pipeline:

```bash
python Code/main.py
```

This script:

* Downloaded NIFTY 50 data via `yfinance`
* Trained ARIMA, SARIMAX, and LSTM models
* Printed next-day Open/Close predictions
* Computed RMSE & MAPE back-test metrics
* Saved results to `forecast_results.csv` and plots `forecast_plot_Open.png` & `forecast_plot_Close.png`

---

## Testing

All automated tests were implemented using pytest. To execute the test suite:

```bash
make test
```

Which runs:

```bash
pytest Code/tests --maxfail=1 --disable-warnings -q
```

Individual test files covered:

* `test_data_handler.py` — data loading & cleaning logic
* `test_config.py` — config utilities
* `test_arima_model.py` — ARIMA walk-forward forecasting & model saving
* `test_lstm_model.py` — LSTM training & prediction logic
* `test_main.py` — end-to-end pipeline smoke test

---

## Continuous Integration

GitHub Actions was configured to run tests on every push and pull request to `main`. The workflow (`.github/workflows/ci.yml`) used Python 3.10:

```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to your fork (`git push origin feature/XYZ`)
5. Open a pull request against `main` and ensure all tests pass

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

from datetime import datetime, timedelta
import pandas as pd

# Number of past time steps (days) to use for prediction
# 60 is chosen to capture approximately 3 months of trading data (~20 trading days per month)
LOOKBACK = 60

# Interval to retrain the model during walk-forward forecasting
# 5 is used to retrain every 5 prediction steps to balance between adaptability and computation time
RETRAIN_INTERVAL = 5

# Size of the rolling training window (in years)
# 3 years offers enough historical context while remaining recent enough to avoid outdated trends
ROLLING_WINDOW_YEARS = 3

def get_train_end_date(df, gap_years=1):
    """
    Calculates the end date of the training data range by subtracting `gap_years` from the max date.
    
    Args:
        df (pd.DataFrame): The input time series DataFrame.
        gap_years (int): The number of years to exclude from training data (for validation or testing).
    
    Returns:
        pd.Timestamp: The calculated training cutoff date.
    """
    return df.index.max() - pd.DateOffset(years=gap_years)


def get_last_weekday():
    """
    Returns the most recent weekday before today (i.e., skips Saturday and Sunday).
    Useful for setting end date in data fetching.
    """
    today = datetime.today()
    offset = 1
    while (today - timedelta(days=offset)).weekday() > 4:  # 5 = Saturday, 6 = Sunday
        offset += 1
    return (today - timedelta(days=offset)).strftime("%Y-%m-%d")
##
# @file config.py
# @brief Configuration constants and helper functions for forecasting models.
#
# @details
# This module defines key hyperparameters such as lookback window, retraining interval,
# and training range limits. It also includes utilities for computing valid training
# cutoff dates and filtering non-weekend days for data fetching.
#
# All constants are documented with their rationale.
# Functions are annotated with Doxygen-compatible docstrings.
#
# @date June 2025

##

from datetime import datetime, timedelta
import pandas as pd

##
# @var lookback
# @brief Number of past time steps (days) used as input for LSTM or sliding window ARIMA models.
#
# @details
# A value of 60 days captures around 3 months of trading activity (assuming ~20 trading days/month).
# This is a common practice in financial modeling to balance temporal context with noise.
##
lookback = 60

##
# @var retrainInterval
# @brief Interval (in days) at which the model is retrained during walk-forward validation.
#
# @details
# Retraining every 5 prediction steps provides adaptability to changing trends
# without incurring excessive computational cost.
##
retrainInterval = 5

##
# @var rollingWindowYears
# @brief Historical window size (in years) for training the forecasting model.
#
# @details
# 3 years is chosen to provide enough historical coverage for stable model learning
# while ensuring relevance to current market patterns.
##
rollingWindowYears = 3

##
# @brief Compute the last valid training date for a given dataset.
#
# @param df The input time series DataFrame with a datetime index.
# @param gap_years Number of years to subtract from the end for model validation or testing.
#                  Default is 1 year.
# @return pd.Timestamp representing the end date of training data.
#
# @details
# Useful for walk-forward validation where the test set is separated from the training set.
##
def getTrainEndDate(df, gap_years=1):
    return df.index.max() - pd.DateOffset(years=gap_years)

##
# @brief Get the most recent weekday before today.
#
# @return str representing the last non-weekend date in YYYY-MM-DD format.
#
# @details
# Ensures data fetching avoids weekends (non-trading days).
# Useful when running scripts automatically or near real-time.
##
def get_last_weekday():
    today = datetime.today()
    offset = 1

    ##
    # Loop backward until a weekday is found
    # Skip Saturday (5) and Sunday (6)
    ##
    while (today - timedelta(days=offset)).weekday() > 4:
        offset += 1

    return (today - timedelta(days=offset)).strftime("%Y-%m-%d")
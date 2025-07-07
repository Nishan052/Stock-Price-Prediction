##
# @file test_config.py
# @brief Unit tests for configuration utility functions in config.py
#
# @details
# These tests validate the behavior of `getTrainEndDate()` for both
# default and custom `gap_years` values.
#
# @date June 2025

##

import pandas as pd
from config import getTrainEndDate

##
# @brief Test `getTrainEndDate` with the default gap of 1 year.
#
# @details
# Builds a dummy DataFrame indexed monthly from Jan to Jun 2021.
# Checks if the function correctly subtracts 1 year from max date.
##
def testGetTrainEndDateDefaultGap():
    # build a simple DateTimeIndex
    dates = pd.date_range("2021-01-01", "2021-06-01", freq="M")
    df = pd.DataFrame(index=dates)

    # default gap_years=1
    result = getTrainEndDate(df)
    expected = df.index.max() - pd.DateOffset(years=1)

    # assert the offset is correct
    assert result == expected

##
# @brief Test `getTrainEndDate` with a custom gap of 2 years.
#
# @details
# Builds a daily DataFrame for 2020 and validates correct subtraction of 2 years.
##
def testGetTrainEndDateCustomGap():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    df = pd.DataFrame(index=dates)

    # custom gap_years=2
    result = getTrainEndDate(df, gap_years=2)
    expected = df.index.max() - pd.DateOffset(years=2)

    assert result == expected
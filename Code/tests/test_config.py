import pandas as pd
from config import get_train_end_date

def test_get_train_end_date_default_gap():
    # build a simple DateTimeIndex
    dates = pd.date_range("2021-01-01", "2021-06-01", freq="M")
    df = pd.DataFrame(index=dates)

    # default gap_years=1
    result = get_train_end_date(df)
    expected = df.index.max() - pd.DateOffset(years=1)
    assert result == expected

def test_get_train_end_date_custom_gap():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    df = pd.DataFrame(index=dates)

    # custom gap_years=2
    result = get_train_end_date(df, gap_years=2)
    expected = df.index.max() - pd.DateOffset(years=2)
    assert result == expected
import pandas as pd
import os


def import_and_wrangle_data(path_to_input):
    """

        Author: Marcel Sibbe, marcel-sibbe@web.de


    Imports options data from a specified parquet file, performs data wrangling, and returns a processed DataFrame.

    Parameters:
        path_to_input (str): The file path to the input parquet file containing options data.

    Returns:
        pd.DataFrame: Processed DataFrame with additional columns for option_id, time to maturity (ttm),
                      and a flag indicating if the date is a Monday (selection_date).

    Raises:
        AssertionError: If the file extension is not '.parquet'.

    This function reads options data from the specified parquet file, adds a unique identifier 'option_id' based on
    symbol, strike, and option expiration, calculates the time to maturity (ttm) in days, and flags Mondays with the
    'selection_date' column. Additionally, it handles missing stock prices by forward-filling with the last available
    value. The resulting DataFrame is ready for further analysis or processing.
    """

    # Get the file extension from the path
    file_extension = os.path.splitext(path_to_input)[-1].lower()

    # Assert that the file has a parquet extension
    assert file_extension == ".parquet", "The input file does not have a parquet extension."

    option_data = pd.read_parquet(path_to_input)

    # Now you can use the DataFrame 'df' for further analysis or processing
    option_data['option_id'] = option_data.apply(
        lambda row: f"{row['symbol']}_{row['strike']}_{row['option_expiration']}", axis=1)
    # number of days to expiry

    option_data["date"] = pd.to_datetime(option_data["date"], format="%Y-%m-%d")
    option_data["option_expiration"] = pd.to_datetime(option_data["option_expiration"], format="%Y-%m-%d")

    option_data["ttm"] = (option_data["option_expiration"] - option_data["date"]).dt.days

    option_data.sort_index(ascending=True, inplace=True)

    # check whether date is monday or not and add flag:

    option_data['selection_date'] = (option_data['date'].dt.dayofweek == 0) * 1

    # In case that there are any missing stock prices -> fill forward with last available, not the best solution, but
    # should be sufficient for the backtest

    option_data["stock_price_close"] = option_data["stock_price_close"].fillna(method='ffill')

    return option_data

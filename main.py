  # 0) Import necessary libraries and modules
import pandas as pd
from ShortPutBacktester import ShortPutBacktester

# 1) Set up settings for the backtester
settings = {
    "total_aum": 100_000_000,  # Assets Under Management of total portfolio in $
    "short_put_bucket": 0.5,  # Pct value of total AUM allocated to short put strategy->
    # only relevant for [inverse_maximum_loss_on_vola_bucket] trade sizing.
    "spx_contract_size": 100,  # Assumption: All options have a contract size of 100
    "target_delta": -0.10,  # find option from daily chain with delta closest to -0.10
    "ttm": 30,  # find option from daily chain with ttm closest to 30 days
    "rolling_exposure": 1/4,  # portion of target size to be invested in a single instrument only relevant
    # for [inverse_maximum_loss_on_vola_bucket] trade sizing
    "trade_sizing": "inverse_maximum_loss_on_vola_bucket",  # select function to be applied to size each trade
    #  ["3pct_maximum_loss_on_total_aum_per_trade", "inverse_maximum_loss_on_vola_bucket"] -> for the first version,
    # you can set the number in the string to apply the percentage threshold in the backtest,
    # e.g. 5pct_maximum_loss_on_total_aum_per_trade would set the threshold to 5% for the backtest.
    "parquet_file_path": r"pathtooptionsdata"
}

# 2) Import option data
option_data = pd.read_parquet(settings["parquet_file_path"])

# 3) Initialize ShortPutBacktester
bt = ShortPutBacktester(settings, option_data)

# 4) Run backtest
results = bt.run_backtest()

print(results["analytics"])

bt.plot_portfolio_performance(results["portfolio_value"])


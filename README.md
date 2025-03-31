# Backtesting a Short-Put Option Premium Strategy  

The strategy selects -10 Delta puts with 30 days to maturity, trading on Mondays. Positions are held until expiration, with profits calculated based on the difference between the initial price and the value at expiration.

## Position Sizing:
- **3% Maximum Loss:** Limits the loss on each position to 3% of the total portfolio value. Average of 3.89 positions held.
- **Inverse Target Size:** Adjusts position size based on the maximum loss and portfolio exposure, with a volatility bucket of 20% and rolling exposure of 25%.

## Features
- ✅ Setting up the parameters of the strategy
- ✅ Running historical simulations using daily SPY option chains  
- ✅ Printing basic analytics to analyze performance  

## Installation
Step-by-step guide to install and set up the project.

```sh
# Clone the repo
git clone https://github.com/RbxCap/ShortPutBacktester.git

# Navigate to the project directory
cd ShortPutBacktester

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

## Usage  
Data comes from a parquet file or directly via the `OptionsDataFetcher` from the **AlphaVantageDataLoader** project.

```python
# 0) Import necessary libraries and modules
import pandas as pd
from ShortPutBacktester import ShortPutBacktester
from config import path_to_data

# 1) Set up settings for the backtester
settings = {
    "total_aum": 100_000_000,  # Total portfolio AUM in $
    "short_put_bucket": 0.2,  # % of AUM allocated to short put strategy
    "spx_contract_size": 100,  # All options have a contract size of 100
    "target_delta": -0.10,  # Find option with delta closest to -0.10
    "ttm": 30,  # Find option with time-to-maturity closest to 30 days
    "rolling_exposure": 1/4,  # Portion of target size in a single instrument
    "trade_sizing": "inverse_maximum_loss_on_vola_bucket",  
    "parquet_file_path": path_to_data
}

# 2) Import option data
option_data = pd.read_parquet(settings["parquet_file_path"])

# 3) Initialize ShortPutBacktester
bt = ShortPutBacktester(settings, option_data)

# 4) Run backtest
results = bt.run_backtest()

print(results["analytics"])

bt.plot_portfolio_performance(results["portfolio_value"])
```

## 📊 Performance Metrics  2021-01-04 to 2025-03-28

| Metric                 | Value     |
|------------------------|----------|
| **Mean Return (ann.)** | 7.64%    |
| **Volatility (ann.)**  | 0.91%    |
| **Total Return**       | 38.01%   |
| **Max Drawdown**       | -2.67%   |
| **Sharpe Ratio**       | 8.38     |
| **Calmar Ratio**       | 2.86     |
| **Avg Positive Return** | 0.0473%  |
| **Avg Negative Return** | -0.0804% |

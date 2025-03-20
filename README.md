# ShortPutBacktester
Backtesting a Short Put Option Premium Strategy


Shorting Monthly SPX Puts Summary

This strategy involves selling monthly -10 Delta SPX put options, profiting from time decay and volatility contraction while taking on the risk of market downturns. The strategy works best in stable markets with low volatility, but carries crash risk during market declines. The analysis period is from January to December 2008, covering the subprime crisis.

Data: Includes 2816 SPX put options for 2008, with daily option chain data.

Methodology: The strategy selects -10 Delta puts with 30 days to maturity, trading on Mondays. Positions are held until expiration, with profits calculated based on the difference between the initial price and the value at expiration.

Position Sizing:

3% Maximum Loss: Limits the loss on each position to 3% of the total portfolio value. Average of 3.89 positions held.
Inverse Target Size: Adjusts position size based on the maximum loss and portfolio exposure, with a volatility bucket of 20% and rolling exposure of 25%.

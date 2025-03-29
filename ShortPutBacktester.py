import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


class ShortPutBacktester:
    """

    Author: Marcel Sibbe, marcel-sibbe@web.de

    Class for backtesting a short put strategy.

    Attributes:
        open_positions (pd.DataFrame): DataFrame to store open positions.
        closed_positions (pd.DataFrame): DataFrame to store closed positions.
        portfolio_value (pd.DataFrame): DataFrame to track portfolio value over time.
        historical_selections (dict): Dictionary to store historical selections (open_position at any given day).
        vola_portfolio_value (float): Amount of total AUM allocated to the short put bucket.
        spx_contract_size (int): Contract size -> assumption for all options = 100. It's constant for this exercise but
        might vary in reality depending on the options used.
        options_data (pd.DataFrame): DataFrame containing options data.

    Methods:
        __init__: Initializes the ShortPutBacktester with settings and options_data.
        update_portfolio_value: Updates the portfolio value based on new information.
        position: Selects instrument and compute trade and adds the new position to open_positions
        run_backtest: Performs the backtest and creates portfolio_values, closed_positions, historical_selections,
        analytics
        update: Updates positions based on daily options data.
        short_put_payoff: Calculates the payoff for a short put position.
        max_loss: Calculates the maximum loss for a short put position.
        find_option: Finds an option in chain that fits to provides requirements such as distance to ttm and delta.
        show_options_of_instrument: User provides a unique option_id and function returns all data for that option_id
        run_analytics: Performs a basic set of analytics to analyse the final portfolio levels -> Assumption, RF=0%
        plot_portfolio_performance: Plots a cumulative return series, max drawdown series and 30 days rolling volatility
    """

    def __init__(self, settings: dict, options_data: pd.DataFrame):
        """
        Initialize the ShortPutBacktester.

        Parameters:
            settings (dict): Dictionary containing various settings for the backtester.
            options_data (pd.DataFrame): DataFrame containing options data.

        Returns:
            None
        """

        # check input data provided

        assert isinstance(settings, dict), "Input settings must be provided as dictionary"
        assert isinstance(options_data, pd.DataFrame), "Options data must be provided as pd.DataFrame"
        assert "type" in options_data.columns, "'type' column is missing in options_data"
        assert (options_data["type"] == "put").sum() > 0, "No puts in options data"

        required_keys = [
            "total_aum",
            "short_put_bucket",
            "spx_contract_size",
            "target_delta",
            "ttm",
            "rolling_exposure",
            "parquet_file_path"
        ]

        for key in required_keys:
            assert settings[key] is not None, f"Value for '{key}' cannot be missing."

        # Create an empty DataFrame with just the column names for the open_positions and closed_positions df's
        column_names = options_data.columns.tolist()
        column_names.extend([
            "current_date", "date_opened", "is_active",
            "pnl", "option", "start_price",
            "units", "income_per_option", "mtm_value"
        ])

        self.settings = settings

        self.open_positions = pd.DataFrame(columns=column_names)
        self.closed_positions = pd.DataFrame(columns=column_names)

        # Initialize portfolio value DataFrame
        self.portfolio_value = pd.DataFrame(
            {'Value': [settings['total_aum']]},
            index=[options_data["date"].iloc[0]]
        )

        self.historical_selections = dict()

        # Calculate value of the short put bucket
        self.vola_portfolio_value = float(self.settings['short_put_bucket'] * self.settings['total_aum'])

        # Set SPX contract size
        self.spx_contract_size = int(self.settings['spx_contract_size'])

        # Set input data
        self.options_data = options_data
        self.start_date = self.options_data["date"].iloc[0]
        self.end_date = self.options_data["date"].iloc[-1]
        self.current_date = None

        # Set pnl metrics
        self.received_premium = 0
        self.loss_expired_position = 0
        self.mtm_pnl = 0

    def update(self, daily_options_data: pd.DataFrame):
        """
        Update portfolio value based on new information.

        Parameters:
            positions (pd.DataFrame): DataFrame containing open positions.
            daily_options_data (pd.DataFrame): DataFrame containing daily options data.

        Returns:
            dict: Dictionary containing updated open positions and closed positions.
        """
        positions = self.open_positions

        positions.loc[:, 'current_date'] = daily_options_data["date"].iloc[0]

        # Sort options_data by option_id
        options_data_sorted = daily_options_data.sort_values(by='option_id')

        # Set option_id as the index for faster lookup
        options_data_sorted.set_index('option_id', inplace=True)

        # Update stock_price_close, bid, ask, mean_price, delta in positions using options_data
        positions.loc[:, 'stock_price_close'] = positions['option_id'].map(options_data_sorted['stock_price_close'])
        positions.loc[:, 'bid'] = positions['option_id'].map(options_data_sorted['bid'])
        positions.loc[:, 'ask'] = positions['option_id'].map(options_data_sorted['ask'])
        positions.loc[:, 'mean_price'] = positions['option_id'].map(options_data_sorted['mean_price'])

        positions.loc[:, 'delta'] = positions['option_id'].map(
            lambda x: float(options_data_sorted['delta'][x][0]) if isinstance(options_data_sorted['delta'][x],
                                                                              list) else float(
                options_data_sorted['delta'][x]))
        positions.loc[:, 'vega'] = positions['option_id'].map(
            lambda x: float(options_data_sorted['vega'][x][0]) if isinstance(options_data_sorted['vega'][x],
                                                                             list) else float(
                options_data_sorted['vega'][x]))
        positions.loc[:, 'gamma'] = positions['option_id'].map(
            lambda x: float(options_data_sorted['gamma'][x][0]) if isinstance(options_data_sorted['gamma'][x],
                                                                              list) else float(
                options_data_sorted['gamma'][x]))
        positions.loc[:, 'theta'] = positions['option_id'].map(
            lambda x: float(options_data_sorted['theta'][x][0]) if isinstance(options_data_sorted['theta'][x],
                                                                              list) else float(
                options_data_sorted['theta'][x]))
        positions.loc[:, 'rho'] = positions['option_id'].map(
            lambda x: float(options_data_sorted['rho'][x][0]) if isinstance(options_data_sorted['rho'][x],
                                                                            list) else float(
                options_data_sorted['rho'][x]))

        # Check if active -> positions is active if option_expiration is after current_date
        positions.loc[:, 'is_active'] = positions['current_date'] < positions['option_expiration']

        # Move expired positions to closed_positions and remove from open_positions
        active_positions = positions[positions["is_active"] == 1]
        expired = positions[positions["is_active"] == 0].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # Compute pnl of closed positions -> itm: (stock_price - strike + option_income), otm: option_income
        if expired["stock_price_close"].isna().any():
            current_stock_price = float(
                self.options_data[self.options_data["date"] == self.current_date]["stock_price_close"].iloc[0]
            )
            expired["stock_price_close"] = expired["stock_price_close"].fillna(current_stock_price)

        for i in range(expired.shape[0]):
            expired["pnl"] = expired["pnl"].astype(float)
            expired.loc[:, "pnl"].iloc[i] = self.short_put_payoff(
                expired["stock_price_close"].iloc[i],
                expired["strike"].iloc[i],
                expired["income_per_option"].iloc[i]
            )

        return {
            "open_positions": active_positions,
            "expired": expired
        }

    def position(self, daily_options_data: pd.DataFrame):
        """
        Selectes and trades a position and adds it to open_positions data frame

        Parameters:
            daily_options_data (pd.DataFrame): DataFrame containing options chain of a single day

        Returns:
            ""
        """

        # Apply find_option function to get option from chain that fits the requirements in terms of delta & ttm most.
        option_to_sell = self.find_option(daily_options_data)

        # reset received_premium before any new trade is done
        self.received_premium = 0

        # Check if the option is already in positions
        option_id = option_to_sell["option_id"]

        # Check if positions["option_id"].values is empty
        if not self.open_positions["option_id"].values.any():
            is_in_position = False
        else:
            # Check if option_id is in positions
            is_in_position = bool(option_id.isin(self.open_positions['option_id']).any())

            # If option is not already part of open_positions, we can trade it and add to our portfolio:
        if not is_in_position:
            # Trade instrument
            option_to_sell["date_opened"] = option_to_sell["date"]
            option_to_sell["start_price"] = option_to_sell["stock_price_close"]
            option_to_sell["pnl"] = 0
            option_to_sell["income_per_option"] = option_to_sell["bid"]  # -> Assumption: we can sell the option at bid
            option_to_sell["max_loss_potential"] = self.max_loss(
                option_to_sell["strike"],
                option_to_sell["income_per_option"]
            )

            # each Monday, whenever there is a trade, we only invest 1/4 of the target size (25%)

            # option_to_sell["units"] = (self.vola_portfolio_value * self.settings["rolling_exposure"]) / (
            #         option_to_sell["max_loss_potential"] * self.spx_contract_size)

            option_to_sell = self.trade_sizing_function(option_to_sell, self.settings["trade_sizing"])

            # Check if self.open_positions is empty
            if self.open_positions.empty:
                self.open_positions = option_to_sell.copy()
            else:
                # Concatenate only if self.open_positions is not empty
                self.open_positions = pd.concat([self.open_positions, option_to_sell], ignore_index=True)

            # once the trade has been done, we can immediately add the received premium to the portfolio_value
            # this will be done later in method: run_backtest(), we only prepare the value already here and add to self.
            self.received_premium = float(
                option_to_sell["income_per_option"].iloc[0] * self.spx_contract_size * option_to_sell["units"].iloc[0])

        return

    @staticmethod
    def short_put_payoff(stock_price, strike, option_income):
        """
        Calculate the payoff for a short put position.

        Parameters:
            - stock_price: Current stock price (float or convertible).
            - strike: Strike price of the put option (float or convertible).
            - option_income: Income received from selling the put option (float or convertible).

        Returns:
            float: Payoff for the short put position.
        """
        try:
            # Extract the first value if input is a numpy array
            if isinstance(stock_price, np.ndarray):
                stock_price = stock_price.item()
            if isinstance(strike, np.ndarray):
                strike = strike.item()
            if isinstance(option_income, np.ndarray):
                option_income = option_income.item()

            # Convert to float if input is a string
            stock_price = float(stock_price)
            strike = float(strike)
            option_income = float(option_income)

            # Compute the payoff
            return option_income if stock_price > strike else stock_price - strike + option_income

        except (ValueError, TypeError) as e:
            print(f"Error in short_put_payoff: {e}")
            return None  # Alternative handling (e.g., return 0 or np.nan)


    @staticmethod
    def max_loss(short_put_strike: float, option_premium_received: float):
        """
        Calculate the maximum loss for a short put position.

        Parameters:
            - short_put_strike (float): Strike price of the short put option.
            - option_premium_received (float): Premium received when selling the put option.

        Returns:
            float: Maximum loss for the short put position.
        """
        short_put_strike = float(short_put_strike.iloc[0])
        option_premium_received = float(option_premium_received.iloc[0])

        # Now perform the subtraction
        return short_put_strike - option_premium_received

    def trade_sizing_function(self, option_to_sell: pd.DataFrame, sizing_function: str):
        """
        Function to size each trade and compute units

        Parameters:
            - option_to_sell (pd.DataFrame): dataframe with relevant option data

        Returns:
            ""
        """
        #
        if sizing_function == "inverse_maximum_loss_on_vola_bucket":
            option_to_sell["units"] = (self.vola_portfolio_value * self.settings["rolling_exposure"]) / (
                    option_to_sell["max_loss_potential"] * self.spx_contract_size)

        #
        if not sizing_function == "inverse_maximum_loss_on_vola_bucket":

            match = re.search(r'(\d+(\.\d+)?)pct', sizing_function)

            if match:
                # Extract the matched numeric part
                numeric_part = match.group(1)

                # Convert the numeric part to a decimal
                max_loss_constraint = float(numeric_part) / 100

            option_to_sell["units"] = (
                    max_loss_constraint * self.settings["total_aum"]
                    / (option_to_sell["max_loss_potential"] * self.spx_contract_size)
            )

        return option_to_sell

    def find_option(self, option_chain: pd.DataFrame):
        """
        Find the option in the given option_chain that is closest to the target delta and target time to maturity (TTM).

        Parameters:
        - option_chain (pd.DataFrame): DataFrame containing option chain data.
        - target_delta (float): Target delta value.
        - target_ttm (int): Target time to maturity in days.

        Returns:
        - pd.DataFrame: Subset of option_chain containing the option closest to the target delta and TTM.
        """

        # Assert checks for input data
        assert isinstance(option_chain, pd.DataFrame), "Input 'option_chain' must be a Pandas DataFrame."

        # pre-checks -> Only put for call_put eligible (given input data has only puts, but this could be different with
        # another dataset)

        option_chain = option_chain[option_chain["type"] == "put"]

        # Calculate the absolute distance to the target TTM for each row
        option_chain["distance_to_target_ttm"] = abs(option_chain["ttm"] - self.settings["ttm"])

        # Find the minimum distance to the target TTM
        min_distance_ttm = option_chain["distance_to_target_ttm"].min()

        # Subset the DataFrame to rows with the minimum distance to the target TTM
        sub_option_chain_ttm = option_chain.loc[option_chain["distance_to_target_ttm"] == min_distance_ttm].copy()

        columns_to_convert = ['delta', 'gamma', 'theta', 'vega', 'rho']

        for col in columns_to_convert:
            sub_option_chain_ttm[col] = pd.to_numeric(sub_option_chain_ttm[col], errors='coerce')

        # Handle NaN values in the specified columns
        sub_option_chain_ttm[columns_to_convert] = sub_option_chain_ttm[columns_to_convert].fillna(0)

        # Calculate the absolute distance to the target delta for each row in the subset
        sub_option_chain_ttm["distance_to_target_delta"] = abs(
            sub_option_chain_ttm["delta"] - self.settings["target_delta"]
        )

        # Find the minimum distance to the target delta in the subset
        min_distance_delta = sub_option_chain_ttm["distance_to_target_delta"].min()

        # Subset the DataFrame to rows with the minimum distance to the target delta
        sub_option_chain = sub_option_chain_ttm.loc[
            sub_option_chain_ttm["distance_to_target_delta"] == min_distance_delta].copy()

        # Drop temporary columns used for calculations
        sub_option_chain = sub_option_chain.drop(columns=["distance_to_target_ttm", "distance_to_target_delta"])

        return sub_option_chain

    def show_options_of_instrument(self, option_id: str):
        """
        Returns all data available for a given option_id.

        Parameters:
            option_id (str): option_id

        Returns:
            options_of_instrument: Dataframe with all data available for provided instrument
        """
        options_of_instrument = self.options_data[self.options_data["option_id"] == option_id]

        return options_of_instrument

    def run_backtest(self):
        """
        Process options data based on a short put strategy.

        Parameters:
            options_data (pd.DataFrame): DataFrame containing options data.

        Returns:
            pd.DataFrame: Filtered DataFrame based on the short put strategy.
        """
        # Create a date range which is based on the provided input data -> Can be of course more flexible by making it
        # possible to let the user define start and end date of the bt, but should be fine for now.
        date_range = pd.date_range(start=self.start_date, end=self.end_date)

        # just needed to clean the output a little bit
        columns_to_drop = ["mark"]

        # Filter out weekends (Saturday and Sunday) -> Keep only weekdays in range
        weekdays = date_range[date_range.weekday < 5]

        # Loop over each date and perform selection steps -> get daily options data, update open_positions, close
        # positions if expired and compute cash flows, check whether its a trading day (Monday), if yes, select option
        # and check whether its already part of the portfolio or not. If not -> add option to position and trade.
        # Update portfolio_value based on income_received from newly trades option, potential loss at expiration, mark-
        # to-market-pnl

        for current_date in weekdays:

            print(f"processing date: {current_date}")

            self.current_date = current_date

            # Get the current option chain
            df = self.options_data[self.options_data["date"] == current_date]

            # only dates included in options_data df are relevant
            if df.empty:
                continue

            self.open_positions.loc[:, "current_date"] = current_date

            pos_update = self.update(df)

            self.open_positions = pos_update["open_positions"]

            self.open_positions.loc[:, "mtm_value"] = (
                                                              self.open_positions["income_per_option"] -
                                                              self.open_positions["mean_price"]
                                                      ) * self.spx_contract_size * self.open_positions["units"]

            self.mtm_pnl = float(self.open_positions["mtm_value"].sum())

            expired = pos_update["expired"]

            self.loss_expired_position = float(
                ((expired["pnl"] * expired["units"] * self.spx_contract_size) * (expired["pnl"] < 0)).sum())

            if expired.empty:
                self.closed_positions = self.closed_positions
            else:
                # Concatenate only if self.open_positions is not empty
                self.closed_positions = pd.concat([self.closed_positions, expired], ignore_index=True)

            self.received_premium = 0

            if df["selection_date"].sum() > 0:
                self.position(df)

            self.historical_selections[current_date.strftime("%Y-%m-%d")] = self.open_positions.drop(
                columns=columns_to_drop
            )

            new_value = self.portfolio_value['Value'].iloc[
                            -1] + self.received_premium + self.loss_expired_position + self.mtm_pnl

            # Add a new row with updated portfolio values to the DataFrame
            self.portfolio_value.loc[current_date] = new_value

            # Update bucket size for short put strategy -> its %short_put_bucket times latest portfolio value.
            self.vola_portfolio_value = new_value * self.settings["short_put_bucket"]

        analytics = self.run_analytics(self.portfolio_value)

        self.closed_positions = self.closed_positions.drop(columns=columns_to_drop)

        return {
            "historical_selections": self.historical_selections,
            "closed_positions": self.closed_positions,
            "portfolio_value": self.portfolio_value,
            "analytics": analytics
        }

    @staticmethod
    def run_analytics(stock_prices: pd.DataFrame):
        """
        Calculate basic return and risk measures for a time series of stock levels.

        Parameters:
            stock_prices (pd.Series or np.ndarray): Time series of stock levels.

        Returns:
            dict: Dictionary containing calculated metrics.
        """
        metrics = {}

        # Calculate daily returns
        daily_returns = stock_prices.pct_change().dropna()

        # Calculate annualized mean return
        mean_return = np.mean(daily_returns) * 252  # Assuming 252 trading days in a year

        # Calculate annualized volatility (risk)
        volatility = np.std(daily_returns) * np.sqrt(252)

        # Calculate total return
        total_return = (stock_prices.iloc[-1] / stock_prices.iloc[0]) - 1

        # Calculate maximum drawdown
        max_drawdown = ((stock_prices / stock_prices.cummax()) - 1).min()

        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        sharpe_ratio = mean_return / volatility

        # Calculate Calmar ratio
        calmar_ratio = mean_return / abs(max_drawdown)

        # Calculate average positive and negative returns
        avg_positive_return = np.mean(daily_returns[daily_returns > 0])
        avg_negative_return = np.mean(daily_returns[daily_returns < 0])

        metrics["Mean Return (ann.)"] = mean_return
        metrics["Volatility (ann.)"] = volatility
        metrics["Total Return"] = total_return
        metrics["Max Drawdown"] = max_drawdown
        metrics["Sharpe Ratio"] = sharpe_ratio
        metrics["Calmar Ratio"] = calmar_ratio
        metrics["Avg Positive Return"] = avg_positive_return
        metrics["Avg Negative Return"] = avg_negative_return

        return pd.DataFrame(metrics).T

    @staticmethod
    def plot_portfolio_performance(portfolio_values: pd.DataFrame, rolling_window=30):
        """
        Plot cumulative returns and maximum drawdown series for a time series of portfolio values.

        Parameters:
            portfolio_values (pd.Series or np.ndarray): Time series of portfolio values.

        Returns:
            None (plots the graphs)
        """
        daily_returns = portfolio_values.pct_change().dropna()

        # Calculate cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod() - 1

        # Calculate maximum drawdown series
        max_drawdown_series = ((portfolio_values / portfolio_values.cummax()) - 1)

        # Calculate rolling volatility
        rolling_volatility = daily_returns.rolling(window=rolling_window).std()

        # Set up the figure and axes
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot cumulative returns on the first panel
        ax1.plot(cumulative_returns, label='Cumulative Returns', color='blue')
        ax1.set_ylabel('Cumulative Returns', color='blue')
        ax1.tick_params('y', colors='blue')
        ax1.set_title('Portfolio Performance Analysis - Cumulative Returns', fontsize=16)
        ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        # Plot maximum drawdown series on the second panel
        ax2.plot(max_drawdown_series, label='Max Drawdown', color='red')
        ax2.set_ylabel('Max Drawdown', color='red')
        ax2.tick_params('y', colors='red')
        ax2.set_title('Portfolio Performance Analysis - Max Drawdown', fontsize=16)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        # Plot rolling volatility on the third panel
        ax3.plot(rolling_volatility, label=f'Rolling Volatility ({rolling_window} days)', color='green')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volatility', color='green')
        ax3.tick_params('y', colors='green')
        ax3.set_title('Portfolio Performance Analysis - Rolling Volatility', fontsize=16)
        ax3.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        # Improve layout
        plt.tight_layout()

        # Show the plot
        plt.show()

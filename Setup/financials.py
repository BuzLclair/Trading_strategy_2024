''' Utility function with finance related tools '''


from tqdm import tqdm

from Setup import pd, np
from Setup.inputs import START_DATE

from Setup.data_setup import MarketData, DATES_MATRIX




### 2. Classes related to financial metrics, i.e returns, ratios...
###############################################################################


class FinancialMetrics:
    ''' Produce the financial metrics from a prices serie '''

    def __init__(self, prices_serie):
        self.data = prices_serie
        self.returns_serie = self.returns()


    def returns(self):
        ''' provides the simple returns of a prices serie '''

        data = self.data.copy()
        ret = data / data.shift(1) - 1
        return ret.dropna()


    def cumulative_returns(self, log=True):
        ''' provides the cumulative returns of a prices serie '''

        cum_returns = np.cumprod(1 + self.returns_serie)
        if log == False:
            return cum_returns.dropna()
        return np.log(cum_returns.dropna())


    def sharpe_ratio(self, risk_free_rate=0):
        ''' returns the sharpe ratio of the prices serie '''

        avg_return = (1 + self.returns_serie.mean())**(252) - 1
        sharpe = avg_return / (self.returns_serie.std(ddof=1)*np.sqrt(252))
        return sharpe


    def returns_table(self):
        ''' poduces a split by month return table '''

        returns = self.data.pct_change()
        monthly_data = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        temp_df = pd.DataFrame({'YEAR': monthly_data.index.year, 'Month': monthly_data.index.month, 'Returns': monthly_data})
        ret_table = temp_df.pivot_table(index='YEAR', columns='Month', values='Returns', aggfunc='sum')
        ret_table.rename(columns={1:'JAN', 2:'FEB', 3:'MAR', 4:'APR', 5:'MAY', 6:'JUN', 7:'JUL', 8:'AUG', 9:'SEP', 10:'OCT', 11:'NOV', 12:'DEC'}, inplace=True)
        ret_table['YTD'] = (1+ret_table.fillna(0)).product(axis=1) - 1
        return (ret_table * 100).round(2) # displayed in %


    def max_drawdown(self):
        ''' returns the max dd array of the serie '''

        max_val = self.data.cummax(axis=0)
        return self.data / max_val - 1







### 3. Classes related to weights computation
###############################################################################


class Weights:
    ''' defines the weights of the strategy given the signals '''

    def __init__(self, signal, frequency='monthly'):
        self.signal = signal.copy(deep=True)
        self.signal *= DATES_MATRIX.loc[DATES_MATRIX.index.isin(self.signal.index)]
        self.__last_day_month = self.signal.groupby(pd.Grouper(freq='M')).tail(1).index
        self.__index_days_diff = self.signal.index.to_series().diff().dt.days
        self.signal.iloc[1:,:].apply(lambda row: self.__util_trade_frequency(row, frequency), axis=1)
        self.signal = self.signal.astype(bool)


    def __util_trade_frequency(self, row, frequency):
        ''' makes weights based on the trading frequency.
            The method uses -trade factor- variable which is equal to 1
            if based on the frequency, the day is a trade day '''

        previous_index = self.signal.index[self.signal.index.get_loc(row.name) - 1]
        if frequency == 'daily':
            return None
        elif frequency == 'weekly':
            trade_factor = (self.__index_days_diff[row.name] > 2)
        elif frequency == 'monthly':
            trade_factor = (row.name in self.__last_day_month)
        if trade_factor == 0:
            self.signal.loc[row.name] = self.signal.loc[previous_index]
        else:
            self.signal.loc[row.name] *= trade_factor


    def __weights_cleaner(self, computed_weights):
        ''' cleans the weights df (make sure sum weights = 1, replace nans by 0 '''

        cleaned_weights = computed_weights.copy(deep=True)
        na_values = self.signal.isnull().all(axis=1)
        cleaned_weights.loc[na_values] = cleaned_weights.loc[na_values].fillna(0)
        cleaned_weights = computed_weights / np.sum(computed_weights, axis=1).values[:, np.newaxis] # to make sure the weights sum to 1
        cleaned_weights = cleaned_weights.fillna(1/np.size(self.signal, axis=1))
        return cleaned_weights


    def equi_weighting(self):
        ''' Strategy designed for buy/sell signal (1 or 0)
        gives an equal weights to all assets for which the signal is buy '''

        self.equi_weights = self.signal / self.signal.sum(axis=1).values[:,np.newaxis]
        return self.__weights_cleaner(self.equi_weights)






### 4. Classes related to backtesting
###############################################################################

# class Backtest:
#     ''' class dedicated to perf calc based on weights and asset prices '''

#     def __init__(self, weights, prices, rebalance=False):
#         self.transaction_costs = 0.003
#         self.rebalance = rebalance
#         self.weights = weights
#         self.weights_array = weights.values
#         self.prices = prices[self.weights.columns]
#         self.wealth = 10000 * np.ones(self.weights.shape[0])
#         self.wealth_repartition = np.ones(self.weights.shape)
#         self.units_per_asset = pd.DataFrame(np.ones(self.weights.shape), index=self.weights.index, columns=self.weights.columns)
#         self.temp_upa = 0
#         self.temp_weights = self.weights.iloc[0,:]
#         self.index = 0
#         self.__order_columns()


#     def __order_columns(self):
#         ''' set the df columns in alphabetical order '''

#         self.units_per_asset = self.units_per_asset.reindex(sorted(self.units_per_asset), axis=1)
#         self.prices = self.prices.loc[self.weights.index].reindex(sorted(self.prices), axis=1).values


#     def __class_cleaning(self):
#         ''' transforms arrays back into df '''

#         self.wealth_repartition = pd.DataFrame(self.wealth_repartition, index=self.weights.index, columns=self.weights.columns)
#         self.units_per_asset = pd.DataFrame(self.units_per_asset, index=self.weights.index, columns=self.weights.columns)
#         self.wealth = pd.Series(self.wealth, index=self.weights.index)


    # def __trade_util(self, wealth_repartition, prev_units_per_asset, prices):
    #     ''' function used to simulate trade in perf calc (change in units per assets) '''

    #     current_wealth_per_asset = prev_units_per_asset * prices
    #     amount_to_trade = wealth_repartition - current_wealth_per_asset # gap between wealth desired and allocated by asset
    #     units_per_asset_to_sell = amount_to_trade * (amount_to_trade <= 0) / prices
    #     wealth_from_sell = -1 * (units_per_asset_to_sell * prices * (1 - self.transaction_costs))
    #     buy_wealth_per_asset = amount_to_trade * (amount_to_trade >= 0)
    #     buy_trade_weights = buy_wealth_per_asset / sum(buy_wealth_per_asset) # percentage of the wealth to reallocate by asset
    #     units_per_asset_bought = buy_trade_weights * sum(wealth_from_sell) / (prices * (1 + self.transaction_costs))
    #     units_per_asset = prev_units_per_asset + units_per_asset_to_sell + units_per_asset_bought
    #     return units_per_asset


#     def __perf_calc_util(self, row):
#         ''' performs the low level operations for the perf calc method '''

#         self.index += 1
#         row_prices = self.prices[self.index,:]
#         row_weights = self.weights_array[self.index,:]
#         row_wealth = np.matmul(self.temp_upa.T, row_prices)
#         if self.rebalance == True or np.array_equal(row_weights, self.temp_weights) == False:
#             row_wealth_rep = row_weights * row_wealth
#             row_upa = self.__trade_util(row_wealth_rep, self.temp_upa, row_prices)
#         else:
#             row_wealth_rep = self.temp_upa * row_prices
#             row_upa = row_wealth_rep / row_prices
#         self.wealth_repartition[self.index,:] = row_wealth_rep
#         self.wealth[self.index] = row_wealth
#         self.temp_upa = row_upa
#         self.temp_weights = row_weights
#         return self.temp_upa


#     def perf_calc(self):
#         ''' compute the wealth attributed to each asset over time '''

#         units_per_a = self.units_per_asset.values
#         self.wealth_repartition[0,:] = self.weights_array[0,:] * self.wealth[0]
#         units_per_a[0,:] = self.wealth_repartition[0,:] / self.prices[0,:]
#         self.temp_upa = units_per_a[0,:]
#         units_per_a[1:,:] = np.apply_along_axis(self.__perf_calc_util, axis=1, arr=units_per_a[1:,:])
#         self.__class_cleaning()
#         return self.wealth


class Backtest:
    ''' class dedicated to perf calc based on weights and asset prices '''

    def __init__(self, weights, prices, rebalance=False):
        self.transaction_costs = 0.0015
        self.weights = weights
        self.prices = prices.loc[self.weights.index][self.weights.columns].values
        self.wealth = 10000 * np.ones(self.weights.shape[0])
        self.temp_upa = self.weights / self.prices
        self.upa = self.temp_upa * self.wealth[:,np.newaxis]


    def __class_cleaning(self):
        ''' transforms arrays back into df '''

        self.units_per_asset = pd.DataFrame(self.upa, index=self.weights.index, columns=self.weights.columns)
        self.wealth = pd.Series(self.wealth, index=self.weights.index)


    def __trade_util(self, wealth_repartition, prev_units_per_asset, prices):
        ''' function used to simulate trade in perf calc (change in units per assets) '''

        current_wealth_per_asset = prev_units_per_asset * prices
        amount_to_trade = wealth_repartition - current_wealth_per_asset # gap between wealth desired and allocated by asset
        units_per_asset_to_sell = amount_to_trade * (amount_to_trade <= 0) / prices
        wealth_from_sell = -1 * (units_per_asset_to_sell * prices * (1 - self.transaction_costs))
        buy_wealth_per_asset = amount_to_trade * (amount_to_trade >= 0)
        buy_trade_weights = buy_wealth_per_asset / sum(buy_wealth_per_asset) # percentage of the wealth to reallocate by asset
        units_per_asset_bought = buy_trade_weights * sum(wealth_from_sell) / (prices * (1 + self.transaction_costs))
        units_per_asset = prev_units_per_asset + units_per_asset_to_sell + units_per_asset_bought
        return units_per_asset


    def perf_calc(self):
        ''' compute the wealth attributed to each asset over time '''

        for nb in range(1,len(self.weights.index)):
            row_weights = self.weights.iloc[nb,:]
            prev_index = self.weights.index[nb-1]
            index = self.weights.index[nb]
            self.wealth[nb] = (self.upa.loc[prev_index] * self.prices[nb,:]).sum()
            self.upa.loc[index] = self.temp_upa.loc[index] * self.wealth[nb]

            if np.array_equal(row_weights, self.weights.iloc[nb-1,:]) == False:
                row_wealth_rep = row_weights * self.wealth[nb]
                self.upa.loc[index] = self.__trade_util(row_wealth_rep, self.upa.loc[prev_index], self.prices[nb,:])

        self.__class_cleaning()
        return self.wealth





### 5. Classes related to strategy management
###############################################################################

class Strategy(Backtest, FinancialMetrics):
    ''' class to handle strategy with perf, returns... '''

    def __init__(self, weights, assets_prices, rebalance=True):
        Backtest.__init__(self, weights, assets_prices, rebalance)
        self.perf = self.perf_calc()
        FinancialMetrics.__init__(self, self.perf)




### 6. Classes related to performance analysis
###############################################################################

class Bootstraping:
    ''' generates random signals based on the investment universe '''

    def __init__(self, signal, prices):
        self.prices = prices
        self.signal = signal
        self.strategy_sharpe = self.strat_sharpe(self.signal)
        self.sharpe_list = []


    def strat_sharpe(self, signal):
        ''' gets strategy signal, returns sharpe ratio '''

        strat_weights = Weights(signal.astype(bool)).equi_weighting()
        strat_wealth = Strategy(strat_weights, self.prices, rebalance=False).wealth
        sharpe = FinancialMetrics(strat_wealth).sharpe_ratio()
        return sharpe


    def random_strat(self):
        ''' create a random strategy signal '''

        random_signal = self.signal.copy(deep=True)
        random_signal.iloc[252:,:] *= np.random.randint(low=0, high=2, size=self.signal.iloc[252:,:].shape)
        return random_signal


    def bootstrap(self, nb_iterr):
        ''' simulate random strat for the number of iterr, returns a list of sharpe ratios '''

        for nb in tqdm(range(nb_iterr)):
            temp_signal = self.random_strat()
            self.sharpe_list.append(self.strat_sharpe(temp_signal))





class VixAnalysis(MarketData):
    ''' analyses perf based on VIX '''

    ticker = ['^VIX']
    vix_data = MarketData(ticker, START_DATE)

    def __init__(self, strat_perf):
        vix_prices = VixAnalysis.vix_data.get_other_prices('Close')
        self.perf = pd.DataFrame({'Strategy':strat_perf, 'VIX':VixAnalysis.vix_data.normalize_data(vix_prices)})
        self.__data_clean()
        self.sharpes = {}
        self.clustering()
        self.sharpe_cluster()


    def __data_clean(self):
        ''' add the returns and cum returns to the perf df '''

        self.perf['returns'] = self.perf['Strategy'].pct_change()
        self.perf['Cumul returns'] = np.cumprod(self.perf['returns'].fillna(0) + 1)


    def clustering(self):
        ''' add cluster categories to the perf df based on indicator '''

        self.perf.loc[(self.perf['VIX'] < 0) & (self.perf['VIX'] < self.perf['VIX'].shift(1)), 'Nowcaster'] = 'Contraction'
        self.perf.loc[(self.perf['VIX'] < 0) & (self.perf['VIX'] >= self.perf['VIX'].shift(1)), 'Nowcaster'] = 'Recovery'
        self.perf.loc[(self.perf['VIX'] >= 0) & (self.perf['VIX'] < self.perf['VIX'].shift(1)), 'Nowcaster'] = 'Slowdown'
        self.perf.loc[(self.perf['VIX'] >= 0) & (self.perf['VIX'] >= self.perf['VIX'].shift(1)), 'Nowcaster'] = 'Expansion'
        self.perf['Nowcaster'] = self.perf['Nowcaster'].shift(1)
        self.perf.dropna(axis=0,how='any',inplace=True)


    def sharpe_cluster(self):
        ''' compute the sharpe ratios based on each VIX period '''

        clustered_returns = self.perf.groupby('Nowcaster')['returns']
        self.sharpes = ((1 + clustered_returns.mean())**252-1)/(np.sqrt(252)*clustered_returns.std())
        self.sharpes = pd.DataFrame(self.sharpes)













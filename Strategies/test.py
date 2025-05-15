from Strategies import u_fin, u_gen, prices_fix

from Strategies.equiweight_buy_and_hold import benchmark


### 0. Strategy class
###############################################################################

#%%

class Nowcaster:
    ''' Class to define the nowcaster strategy
        Bets on the market cycle, trying to capture when the returns are
        quickly coming back up are quickly increasing (and opposite) '''

    def __init__(self, prices):
        self.prices = prices
        self.signal = prices.copy(deep=True)


    def __utils_rolling(self, prices):
        ''' shortcut for prices.rolling(window).mean() '''

        rolled_prices = {'year': prices.rolling(252).mean(), 'quarter': prices.rolling(63).mean(),
                         'month': prices.rolling(20).mean()}
        return rolled_prices


    def global_shape(self):
        ''' says if the assets is overall rising or decreasing '''

        rolled_prices = self.__utils_rolling(self.prices)
        signal = (rolled_prices['quarter'] >= rolled_prices['year']) * (rolled_prices['month'] >= rolled_prices['quarter'])
        signal += (rolled_prices['quarter'] < rolled_prices['year']) * (rolled_prices['month'] >= rolled_prices['month'].shift(20))
        return (signal==1)


    def steering_speed(self):
        ''' says if growth / decrease is getting quicker or slower '''

        rolling_std = self.prices.rolling(10).std()
        rolled_std = self.__utils_rolling(rolling_std)
        signal = (rolled_std['quarter'] >= rolled_std['year']) * (rolled_std['month'] >= rolled_std['month'].shift(1))
        signal += (rolled_std['quarter'] < rolled_std['year']) * (rolled_std['month'] >= rolled_std['month'].shift(10))
        return (signal==1)


    def trade_signal(self):
        ''' return the buy / sell signal (0 or 1) '''

        global_shape_matrix = self.global_shape()
        self.signal = (self.steering_speed()*global_shape_matrix).shift(1).fillna(False)
        all_zeros = self.signal[self.signal.eq(False).all(axis=1)].index
        self.signal.loc[all_zeros] = global_shape_matrix.loc[all_zeros]
        return (self.signal==1)



### 1. Nowcasting strategy
###############################################################################

nowcasting_strat = {}

nowcasting_strat['signal'] = Nowcaster(prices_fix).trade_signal()


nowcasting_strat['weights'] = u_fin.Weights(nowcasting_strat['signal'], frequency='monthly').equi_weighting()
nowcasting_strat['strategy'] = u_fin.Strategy(nowcasting_strat['weights'], prices_fix, rebalance=False)


# sp_benchmark_ret = u_fin.SerieStats(u_fin.MarketData('^GSPC', START_DATE).close_prices).cumulative_returns()
# strategy_plot.multi_plot([sp_benchmark_ret], legends=['S&P assets based strategy','S&P 500'], title='Strategy vs S&P 500 cumulative log returns')



### 2. Performance analysis
###############################################################################

if __name__ =='__main__':

    ########## 2.1 Return plot
    nowcasting_strat['plot'] = u_gen.Plotter(nowcasting_strat['strategy'].cumulative_returns())
    nowcasting_strat['plot'].multi_plot([benchmark['strategy'].cumulative_returns()], legends=['Strategy','Buy & Hold'])


    ########## 2.2 Sharpe ratio
    print(u_fin.FinancialMetrics(nowcasting_strat['strategy'].wealth).sharpe_ratio())


    ########## 2.3 Returns by month
    # u_gen.Heatmaps(u_fin.FinancialMetrics(nowcasting_strat['strategy'].wealth).returns_table()).plotter()


    ########## 2.4 VIX analysis
    # nowcasting_strat['vix'] = u_gen.VixPlot(nowcasting_strat['strategy'].wealth)
    # nowcasting_strat['vix'].visual_returns()
    # nowcasting_strat['vix'].visual_sharpe(benchmark['strategy'].wealth)


    ########## 2.5 Boostrap plotting
    # nowcasting_strat['bootstrap'] = u_fin.Bootstraping(nowcasting_strat['signal'], prices_fix)
    # nowcasting_strat['bootstrap'].bootstrap(100)
    # u_gen.DistribPlot(nowcasting_strat['bootstrap'].sharpe_list, nowcasting_strat['bootstrap'].strategy_sharpe).plotter()



# z = nowcasting_strat['strategy'].returns_serie - benchmark['strategy'].returns_serie

# z1 = z*(z<0)

# import matplotlib.pyplot as plt
# plt.plot(np.cumprod(1+z1))
# plt.tight_layout()
# plt.show


z = nowcasting_strat['weights']





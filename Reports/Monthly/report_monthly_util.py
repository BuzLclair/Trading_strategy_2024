import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import itertools
import random
import seaborn as sns
from scipy import stats
import matplotlib.colors as mcolors

from Setup.data_setup import prices_fix, MarketData, __missing_data
from Setup.financials import Backtest, FinancialMetrics, Strategy
from Reports import nowcasting_strat, benchmark, np



def get_portfolio_file():
    ''' loads the portfolio excel and cleans it '''

    def _get_file_path():
        ''' gets the path to the portfolio excel file '''

        current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        parent_dir0 = os.path.abspath(os.path.join(parent_dir, os.pardir))
        allocation_path = os.path.join(parent_dir0, 'Portfolio.xlsx')
        return allocation_path

    path = _get_file_path()
    allocation_file = pd.read_excel(path, header=1, index_col=0)

    def _clean_portfolio_file(file):
        ''' cleans the loaded portfolio excel file '''

        file['Date'].fillna(method='ffill', inplace=True)
        file['Date'] = pd.to_datetime(file['Date'], format='%d.%m.%Y')
        file.set_index(['Date','Variable'], inplace=True)
        file.drop('Total', axis=1, inplace=True)
        return file

    cleaned_portfolio_data = _clean_portfolio_file(allocation_file)
    return cleaned_portfolio_data



class PortfolioData:
    ''' contains the first level data from the portfolio (dates, simulated portfolios based on period...) '''

    def __init__(self, strategy_w, benchmark_w, save_path):
        self.strategy_w = strategy_w
        self.benchmark_w = benchmark_w
        self.save_path = save_path

        self.file = get_portfolio_file()
        self.dates_list = self.dates_meta()['dates_list']
        self.prices = prices_fix
        self.sp_data = MarketData('^GSPC', '2020-01-01').get_other_prices('Adj Close')

        self.last_period_prices = self.prices.loc[(self.prices.index >= self.dates_list[-2]) & (self.prices.index <= self.dates_list[-1])]
        self.last_period_prices.drop(['ATVI'], axis=1, inplace=True)
        self.random_returns = self.random_returns_distribution()

        self.implem_date = pd.to_datetime('02.07.2023', format='%d.%m.%Y')
        self.implem_prices = self.prices.loc[self.prices.index > self.implem_date]
        self.strategy_weights = self.active_weights()
        self.active_perf, self.new_benchmark_perf, self.no_rebalance_perf = self.active_perf_calc()
        self.performance_wealth = self.perfs_wealth()


    def dates_meta(self):
        ''' gives the date of the last trading period '''

        date_dic = {}
        dates = self.file.index.get_level_values('Date').unique().sort_values(ascending=True)
        date_dic['range'] = f'{dates[-2].strftime("%d.%m.%Y")} to {dates[-1].strftime("%d.%m.%Y")}'
        date_dic['period'] = dates[-1].strftime('%B %Y') if dates[-1].day > 14 else (dates[-1]-pd.DateOffset(months=1)).strftime('%B %Y')
        date_dic['period2'] = datetime.strptime(date_dic['period'], "%B %Y").strftime("%Y.%m")
        date_dic['business days'] = len(pd.bdate_range(start = dates[-2], end= dates[-1]))
        if date_dic['business days'] > 30:
            date_dic['period'] = dates[-2].strftime('%B %Y') + 'to ' + dates[-1].strftime('%B %Y')
        starting_date = pd.to_datetime('2023-07-02')
        date_dic['months since live'] = (dates[-1].year - starting_date.year)*12 + (dates[-1].month - starting_date.month) + 1
        date_dic['dates_list'] = dates
        return date_dic


    def last_period_rescale(self, serie):
        ''' rescale the index of the given serie to be last period only '''

        return serie.loc[(serie.index < self.dates_list[-1]) & (serie.index >= self.dates_list[-2])]


    def perfs_wealth(self):
        ''' compute the perf of strat and benchmark '''

        strat = Backtest(self.strategy_w, self.prices, rebalance=False).perf_calc()
        strat = strat.loc[(strat.index >= '01.01.2020') & (strat.index < self.dates_list[-1])]
        temp_perf = (self.active_perf / self.active_perf[0]) * strat.loc[strat.index <= self.implem_date][-1]
        strat = pd.concat([strat.iloc[strat.index < self.implem_date], temp_perf])
        bench = Backtest(self.benchmark_w, self.prices, rebalance=False).perf_calc()
        bench = bench.loc[(bench.index >= '01.01.2020') & (bench.index < self.dates_list[-1])]
        temp_perf = (self.new_benchmark_perf / self.new_benchmark_perf[0]) * bench.loc[bench.index <= self.implem_date][-1]
        bench = pd.concat([bench.iloc[bench.index < self.implem_date], temp_perf])
        perf = pd.DataFrame({'Strategy': strat, 'Benchmark': bench})
        return perf.dropna()


    def active_weights(self):
        ''' computes the weights of the strategy based on the portfolio excel file's weights '''

        strategy_weights = self.file.loc[self.file.index.get_level_values('Variable') == 'Allocated wealth']
        strategy_weights = strategy_weights.reset_index(level='Variable', drop=True)
        strategy_weights = strategy_weights / strategy_weights.sum(axis=1).values[:,np.newaxis]
        return strategy_weights


    def stop_end_of_period(self, serie):
        return serie.loc[serie.index <= self.dates_list[-1]]


    def active_perf_calc(self):
        ''' computes the returns of the strategy based on the portfolio excel file's weights '''

        active_weights = self.strategy_weights.reindex(self.implem_prices.index, method='ffill')
        active_perf = Backtest(active_weights, self.implem_prices, rebalance=False).perf_calc()
        new_benchmark_weights = self.benchmark_w.reindex(self.implem_prices.index, method='ffill')
        new_benchmark_perf = Backtest(new_benchmark_weights, self.implem_prices, rebalance=False).perf_calc()
        no_rebalance_weights = self.strategy_weights.loc[self.strategy_weights.index < self.dates_list[-2]]
        no_rebalance_weights = no_rebalance_weights.reindex(self.implem_prices.index, method='ffill')
        no_rebalance_perf = Backtest(no_rebalance_weights, self.implem_prices, rebalance=False).perf_calc()
        return self.stop_end_of_period(active_perf), self.stop_end_of_period(new_benchmark_perf), self.stop_end_of_period(no_rebalance_perf)


    def random_returns_distribution(self, sample=30000):
        ''' simulates a sample of returns that could have been drawn in the investment universe '''

        def _signal_to_weight(signal_tuple):
            ''' from a tuple of 0 or 1 values (signal), returns an equiweight list '''

            signal = list(signal_tuple)
            if sum(signal) == 0:
                return [1/len(signal) for x in signal]
            return [x / sum(signal) for x in signal]

        prices = np.array(self.last_period_prices)
        sample = min(sample, len(list(itertools.product([0, 1], repeat=prices.shape[1]))))


        def _random_portfolio_composition(sample):
            ''' from a tuple of 0 or 1 values (signal), returns an equiweight list '''

            signal_combinations = list(itertools.product([0, 1], repeat=prices.shape[1]))
            random_sample = random.sample(signal_combinations, sample)
            weights_combinations = map(lambda x: _signal_to_weight(x), random_sample)
            units_per_asset = map(lambda x: 10000 * np.array(x) / prices[0], weights_combinations)
            return units_per_asset


        units_per_asset = _random_portfolio_composition(sample)
        wealth = map(lambda x: np.sum(x * prices, axis=1), units_per_asset)
        returns = map(lambda x: np.diff(x) / x[:-1], wealth)
        return list(returns)


    def returns_ranking(self):
        ''' returns the ranking of the period return that is given in the first page of the report '''

        monthly_ret = self.active_perf.pct_change().resample('M').apply(lambda x: (1 + x).prod() - 1)
        month_calc = list(map(lambda x: abs(x - self.dates_list[-1]), monthly_ret.index))
        current_month = month_calc.index(min(month_calc))
        monthly_ret_to_period = monthly_ret[:current_month+1]
        ranking = monthly_ret_to_period.rank(ascending=False)
        return int(ranking[-1])


    def eco_cdt_plot(self):
        ''' plot the first page S&P graph of strqtegy period '''

        last_sp_data = self.sp_data.loc[(self.sp_data.index < self.dates_list[-1])][-100:]
        last_period_sp = last_sp_data[(last_sp_data.index>=self.dates_list[-2]) & (last_sp_data.index<=self.dates_list[-1])]

        figure = GraphicsManager(title=None, max_xaxis_ticks=5, max_yaxis_ticks=5, hidden_spines=['top','right','left'])

        figure.ax.plot(last_sp_data.index, last_sp_data, linewidth=1.5, color='#808080')
        figure.ax.plot(last_period_sp.index, last_period_sp, linewidth=2.5, color='#2F5496', label='Factsheet coverage period')
        figure.ax.axhline(y=last_sp_data.mean(), color='black', linestyle='--', linewidth=0.75)
        figure.add_text_box('Average', last_sp_data.index[20], last_sp_data.mean(), text_color='#808080', edge_color='none')
        figure.ax.fill_between(last_period_sp.index, last_period_sp, color='#8CB3E5', alpha=0.3)
        figure.add_text_box(self.dates_list[-2].strftime('%d.%m.%y'), last_period_sp.index[0], min(last_sp_data), text_color='#808080', edge_color='none')
        figure.add_text_box(self.dates_list[-1].strftime('%d.%m.%y'), last_period_sp.index[-1], min(last_sp_data), text_color='#808080', edge_color='none')
        figure.ax.set_ylim([min(last_sp_data),max(last_sp_data)])

        figure.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%y'))
        figure.ax.legend(loc='lower center', fontsize=13, ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.18), labelcolor='#808080')
        figure.ax.yaxis.set_ticks_position('none')
        figure.ax.tick_params(axis='x', which='major', bottom=True, top=False, length=5, width=1.25)
        figure.ax.spines['bottom'].set_color('#808080')
        figure.ax.spines['bottom'].set_linewidth(1)
        figure.ax.tick_params(axis='x', colors='#808080')
        plt.yticks([])

        figure.ax.set_title('S&P 500 value over the factsheet period', fontsize=18, fontname='Arial', color='#808080')
        figure.end_plot(self.save_path, file_title='sp_eco_cdt')


# z = PortfolioData(nowcasting_strat['weights'], benchmark['weights'], save_path='C:/Users/const/OneDrive/Documents/Code/Python/Cresous_v2/Reports/Ressources/Graphics')



#%%

class GraphicsManager:

    def __init__(self, figsize=(9,5), margins=(0.01, 0.005), title=None, max_xaxis_ticks=None, max_yaxis_ticks=None, hidden_spines=['top','right'], x_label='', y_label=''):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=600)
        self.formatting(title, margins, max_xaxis_ticks, max_yaxis_ticks, hidden_spines, x_label, y_label)


    def formatting(self, title, margins, max_xaxis_ticks, max_yaxis_ticks, hidden_spines, x_label, y_label):
        ''' performs the formatting customizations '''

        def _format_axis():
            ''' formats the graph axis '''

            self.ax.xaxis.set_major_locator(plt.MaxNLocator(max_xaxis_ticks))
            self.ax.yaxis.set_major_locator(plt.MaxNLocator(max_yaxis_ticks))
            self.ax.tick_params(axis='both', which='major', labelsize=13)

            self.ax.spines[hidden_spines].set_color('none')
            non_hidden_spines = list(filter(lambda x: x not in hidden_spines, ['top','right','bottom','left']))
            self.ax.spines[non_hidden_spines].set_linewidth(1.25)
            self.ax.set_xlabel(x_label, color='black')
            self.ax.set_ylabel(y_label, color='black')

        def _format_plot_area():
            ''' formats the graph plotting area '''

            self.ax.set_facecolor('white')
            self.ax.grid(False)
            self.ax.margins(x=margins[0], y=margins[1])

        _format_axis()
        _format_plot_area()
        self.ax.set_title(title, fontsize=18, fontname='Arial')


    def add_text_box(self, text, x_coord, y_coord, text_color='black', edge_color='black'):
        ''' method to add a text box to the graph '''

        t = self.ax.text(x_coord, y_coord, text, fontsize=13, ha='center', va='center', color=text_color)
        t.set_bbox(dict(facecolor='white', boxstyle='round, pad=0.2, rounding_size=0.5', edgecolor=edge_color))


    def end_plot(self, save_path, file_title):
        ''' finishes the plot with showing and saving '''

        self.fig.tight_layout()
        self.fig.savefig(f'{save_path}/{file_title}.png', dpi=600, transparent=True, bbox_inches='tight')
        # plt.show()



class ReturnData(PortfolioData):
    ''' class dedicated to the return page of the report (draw graphs etc) '''

    def __init__(self, portfolio_instance):
        self.portfolio_instance = portfolio_instance
        self.key_numbers = self.key_numbers_calc()


    def reboot_perf(self, perf):
        return (perf / perf[0]) * 100


    def key_numbers_calc(self):
        ''' computes key stats of the performance over the last period '''

        monthly_active_ret = self.portfolio_instance.active_perf.pct_change().resample('M').apply(lambda x: (1 + x).prod() - 1)
        month_calc = list(map(lambda x: abs(x - self.portfolio_instance.dates_list[-1]), monthly_active_ret.index))
        current_month = month_calc.index(min(month_calc))
        current_monthly_ret = monthly_active_ret[current_month]

        last_period_perf = self.portfolio_instance.last_period_rescale(self.portfolio_instance.active_perf)
        last_period_perf = self.reboot_perf(last_period_perf)
        key_numbers = {'return': current_monthly_ret, 'pct profitable':sum(last_period_perf>last_period_perf.shift(1))/len(last_period_perf),
                       'high':max(last_period_perf), 'low':min(last_period_perf)}
        return key_numbers


    def strat_compare_plot(self, active_strategy, comparison_strategy, labels_list, title):
        ''' returns a plot comparing the 2 strategies given as inputs over the last 100 days (or less if period available is shorter)
            in the labels list, the active name should be given first '''

        if len(active_strategy) > 100:
            comparison_strategy = comparison_strategy[-100:]
            active_strategy = active_strategy[-100:]

        comparison_strategy = self.reboot_perf(comparison_strategy)
        active_strategy = self.reboot_perf(active_strategy)
        comparison_strategy_last_period = comparison_strategy[(comparison_strategy.index>=self.portfolio_instance.dates_list[-2]) & (comparison_strategy.index<=self.portfolio_instance.dates_list[-1])]
        active_strategy_last_period = active_strategy[(active_strategy.index>=self.portfolio_instance.dates_list[-2]) & (active_strategy.index<=self.portfolio_instance.dates_list[-1])]

        figure = GraphicsManager(title=None, max_xaxis_ticks=6, max_yaxis_ticks=6, hidden_spines=['top','right','left'])

        figure.ax.plot(comparison_strategy.index, comparison_strategy, linewidth=1.5, color='#808080', label=labels_list[1])
        figure.ax.plot(active_strategy.index, active_strategy, linewidth=1.5, color='#2F5496', label=labels_list[0])
        figure.ax.plot(active_strategy_last_period.index, active_strategy_last_period, linewidth=2.5, color='#2F5496', label=f'{labels_list[0]} (last period)')
        figure.add_text_box('Profit line', comparison_strategy.index[0], 100, text_color='#808080', edge_color='none')
        figure.ax.fill_between(active_strategy_last_period.index, active_strategy_last_period, comparison_strategy_last_period, where=(comparison_strategy_last_period > active_strategy_last_period), color='#D21F3C', alpha=0.3)
        figure.ax.fill_between(active_strategy_last_period.index, active_strategy_last_period, comparison_strategy_last_period, where=(comparison_strategy_last_period < active_strategy_last_period), color='#4C9A2A', alpha=0.3)
        figure.add_text_box(active_strategy_last_period.index[0].strftime('%d.%m.%y'), active_strategy_last_period.index[0], min(min(comparison_strategy), min(active_strategy)), text_color='black', edge_color='none')
        figure.add_text_box(active_strategy_last_period.index[-1].strftime('%d.%m.%y'), active_strategy_last_period.index[-1], min(min(comparison_strategy), min(active_strategy)), text_color='black', edge_color='none')

        figure.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%y'))
        figure.ax.legend(loc='lower center', fontsize=13, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.18), labelcolor='#808080')
        figure.ax.yaxis.set_ticks_position('none')
        figure.ax.tick_params(axis='x', which='major', bottom=True, top=False, length=5, width=1.25)
        figure.ax.spines['bottom'].set_color('#808080')
        figure.ax.spines['bottom'].set_linewidth(1)
        figure.ax.tick_params(axis='both', colors='#808080')
        figure.ax.axhline(y=100, color='black', linestyle='--', linewidth=1.25)

        figure.ax.set_title(title, fontsize=18, fontname='Arial', color='#808080')
        figure.end_plot(self.portfolio_instance.save_path, file_title=title)


    def strat_vs_bench_perf(self):
        ''' plot the 100d strategy vs benchmark comparison graph '''

        self.strat_compare_plot(self.portfolio_instance.active_perf, self.portfolio_instance.new_benchmark_perf, ['Strategy', 'Buy and hold'], 'Portfolio performance - Strategy vs Buy and Hold')


    def strat_vs_no_eom_rebalance_perf(self):
        ''' plot the 100d strategy vs strategy without EOM rebalance comparison graph '''

        self.strat_compare_plot(self.portfolio_instance.active_perf, self.portfolio_instance.no_rebalance_perf, ['Strategy', 'Strategy without EOM rebalance'], 'Portfolio performance - Strategy vs Strategy without EOM rebalance')


    def returns_value_plot(self):
        ''' plot the returns in a graph representing the value of $100 invested, strat vs benchmark '''

        data_perf = self.portfolio_instance.performance_wealth.copy()
        data_perf['Strategy'] = (data_perf['Strategy'] / data_perf['Strategy'][0]) * 100
        data_perf['Benchmark'] = (data_perf['Benchmark'] / data_perf['Benchmark'][0]) * 100
        figure = GraphicsManager(title='Value of $100 invested since January 2020', max_xaxis_ticks=8)
        figure.ax.plot(data_perf.index, data_perf['Strategy'], linewidth=1.5, color='#2F5496', label='Strategy')
        figure.ax.plot(data_perf.index, data_perf['Benchmark'], linewidth=1.5, color='#808080', label='Buy and hold')
        figure.ax.axhline(y=100, color='black', linestyle='--', linewidth=1.25)

        figure.ax.fill_between(data_perf.index, min(data_perf.min()), max(data_perf.max()), where=(data_perf.index <= self.portfolio_instance.implem_date), color='#808080', alpha=0.1, edgecolor='none')

        figure.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%y'))
        figure.ax.legend(loc='lower center', fontsize=13, ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.18))
        figure.ax.yaxis.set_ticks_position('none')
        figure.ax.tick_params(axis='x', which='major', bottom=True, top=False, length=5, width=1.25)
        figure.end_plot(self.portfolio_instance.save_path, file_title='strategy_investment_value')


    def last_sharpe(self):
        ''' computes the sharpe ratio for the last period '''

        last_period_perf = self.portfolio_instance.last_period_rescale(self.portfolio_instance.active_perf)
        last_period_sharpe = FinancialMetrics(last_period_perf).sharpe_ratio()
        return last_period_sharpe


    def sharpe_distribution(self):
        ''' simulates a sample of returns that could have been drawn in the investment universe '''

        sharpes = map(lambda x: ((1 + x.mean())**252 -1) / (x.std(ddof=1)*np.sqrt(252)), self.portfolio_instance.random_returns)
        return list(sharpes)


    def returns_distrib_plot(self):
        ''' plot the random sampling sharpe distrib '''

        sharpe_distrib = self.sharpe_distribution()

        def sharpe_prepare():
            last_period_sharpe = self.last_sharpe()
            last_period_bench = self.portfolio_instance.last_period_rescale(self.portfolio_instance.new_benchmark_perf)
            sharpe_bench = FinancialMetrics(last_period_bench).sharpe_ratio()
            last_period_no_reb = self.portfolio_instance.last_period_rescale(self.portfolio_instance.no_rebalance_perf)
            sharpe_no_reb = FinancialMetrics(last_period_no_reb).sharpe_ratio()
            return last_period_sharpe, sharpe_bench, sharpe_no_reb

        last_period_sharpe, sharpe_bench, sharpe_no_reb = sharpe_prepare()
        percentile_strat = round(stats.percentileofscore(sharpe_distrib, last_period_sharpe), 1)

        figure = GraphicsManager(title='Sharpe ratio, strategy compared to random sample', max_xaxis_ticks=10, x_label='Sharpe ratio', y_label='Density')
        sns.histplot(sharpe_distrib, bins=20, stat='density', color='#2F5496', alpha=0.5)
        sns.kdeplot(sharpe_distrib, color='#808080')
        figure.ax.axvline(x=last_period_sharpe, color='black', linestyle='--', linewidth=2, label='Strategy')
        figure.ax.axvline(x=sharpe_bench, color='#808080', linestyle='--', linewidth=2, label='Buy and hold')
        figure.ax.axvline(x=sharpe_no_reb, color='#808080', linestyle='-.', linewidth=2, label='Strategy without EOM rebalance')
        figure.add_text_box(text=f'Percentile\n{percentile_strat}%', x_coord=last_period_sharpe, y_coord=0.1)
        figure.ax.tick_params(axis='both', colors='black')
        figure.ax.yaxis.set_ticks_position('none')
        figure.ax.legend(loc='lower center', fontsize=13, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.25))
        figure.end_plot(self.portfolio_instance.save_path, file_title='strategy_returns_ditrib')


    def monthly_returns_table(self, df=False):
        ''' return the monthly returns table '''

        fin_metrics_instance = FinancialMetrics(self.portfolio_instance.performance_wealth['Strategy'])
        returns_df = fin_metrics_instance.returns_table().fillna('').iloc[-4:,:]
        if df != False:
            return returns_df
        returns_array_prep = returns_df.reset_index()
        returns_array_prep = returns_array_prep.T.reset_index().T
        returns_array_prep.iloc[1:,0] = returns_array_prep.iloc[1:,0].astype(int)

        return np.array(returns_array_prep)



# z = strategy_data
# z1 = RiskData(z)
# z1.std_risk_plot()





class RiskData(PortfolioData):
    ''' class dedicated to the risk part of the report (draw graphs etc) '''

    def __init__(self, portfolio_instance):
        self.portfolio_instance = portfolio_instance
        self.risk_data = pd.DataFrame({'Strategy':self.portfolio_instance.performance_wealth['Strategy']})
        self.risk_data['returns'] = self.risk_data['Strategy'].pct_change()
        self.std_risk = self.std_calc()
        self.drawdown = self.dd_calc()
        self.last_period_std = self.last_std()


    def std_calc(self):
        ''' method needed to compute the std data for the risk indicator graph '''

        weekly_risk = pd.DataFrame({'Risk': np.sqrt(252) * self.risk_data['returns'].resample('W').std()})
        etoro_bins = [-float('inf'), 0.03, 0.06, 0.09, 0.125, 0.16, 0.21, 0.27, 0.34, 0.43, 0.55, float('inf')]
        etoro_risk_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        weekly_risk['Risk indic'] = pd.cut(weekly_risk['Risk'], bins=etoro_bins, labels=etoro_risk_labels, right=False).astype(float)
        monthly_risk = pd.DataFrame({'Risk':self.portfolio_instance.performance_wealth['Strategy'].pct_change().resample('M').std() * np.sqrt(252)}).iloc[-12:,:].dropna() # monthly data keep YTD only
        monthly_risk['Month'] = monthly_risk.index.strftime('%b') + monthly_risk.index.strftime('%y')
        monthly_risk['Risk indic'] = weekly_risk['Risk indic'].resample('M').mean()
        return monthly_risk


    def std_risk_plot(self):
        ''' method to plot the std data for the risk indicator graph '''

        colors = [(0, 0.5, 0), (0.4, 0.8, 0.4), (1, 1, 0.1), (0.8, 0.4, 0.4), (0.5, 0, 0)]
        n_bins = np.linspace(0, 1, len(colors))
        cmap = mcolors.LinearSegmentedColormap.from_list('', list(zip(n_bins, colors)))
        color_palette = [cmap(val/10) for val in self.std_risk['Risk indic']]

        figure = GraphicsManager(title='Annualized volatility by month', max_xaxis_ticks=8, margins=(0, 0.05), hidden_spines=['top','right','left'])
        sns.barplot(data=self.std_risk, x='Month', y='Risk', palette=color_palette)
        for i in figure.ax.containers:
            figure.ax.bar_label(i,['{:.1f}%'.format(100*x) for x in self.std_risk['Risk']], fontsize=13, fontname='Arial')

        figure.ax.tick_params(axis='x', which='major', labelsize=13)
        plt.yticks([])
        plt.ylabel('')
        plt.xlabel('')
        figure.ax.tick_params(axis='x', colors='black')
        figure.ax.yaxis.set_ticks_position('none')
        figure.end_plot(self.portfolio_instance.save_path, file_title='strategy_std_risk')


    def dd_calc(self):
        ''' method needed to compute the drawdown data for its graph '''

        annual_data = pd.DataFrame(self.portfolio_instance.performance_wealth).iloc[-252:,:].dropna()
        drawdown = FinancialMetrics(annual_data).max_drawdown()
        return drawdown


    def dd_plot(self):
        ''' method to plot the drawdown graph '''

        figure = GraphicsManager(title='Performance drawdown - Last 12 months', max_xaxis_ticks=8, max_yaxis_ticks=7, margins=(0.01,0), hidden_spines=['top','right','top'])

        figure.ax.plot(self.drawdown.index, self.drawdown['Strategy'], linewidth=1.5, color='#2F5496', label='Strategy')
        figure.ax.plot(self.drawdown.index, self.drawdown['Benchmark'], linewidth=1.5, color='#808080', label='Buy and hold')
        figure.ax.axhline(y=0, color='black', linestyle='--', linewidth=1.25)

        figure.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%y'))
        figure.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        figure.ax.legend(loc='lower center', fontsize=13, ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.18))
        figure.ax.yaxis.set_ticks_position('none')
        figure.ax.tick_params(axis='x', which='major', bottom=True, top=False, length=4, width=1.25)
        figure.ax.xaxis.tick_top()
        figure.end_plot(self.portfolio_instance.save_path, file_title='drawdown_risk_plot')


    def beta_calc(self):
        ''' compute beta from returns df with one column strategy returns and the other benchmark '''

        ret = pd.DataFrame({'Strategy':self.portfolio_instance.performance_wealth['Strategy']})
        ret['Market'] = self.portfolio_instance.sp_data.loc[ret.index]
        ret = (ret / ret.shift(1) - 1).dropna()
        rolling_corr = ret.rolling(252).corr(ret, pairwise=True).unstack(1).iloc[:,0]
        rolling_var = ret.rolling(252).var()
        return (rolling_corr * (rolling_var.iloc[:,0] / rolling_var.iloc[:,1])).dropna()


    def beta_plot(self):
        ''' method to plot the rolling beta '''

        beta = self.beta_calc()
        avg_beta = np.mean(beta)

        figure = GraphicsManager(title='Yearly Rolling Beta (using S&P 500 as market proxy)', max_xaxis_ticks=8, max_yaxis_ticks=7, hidden_spines=['top','right','left'])

        figure.ax.plot(beta.index, beta, linewidth=1.5, color='#2F5496', label='Yearly Rolling Beta')
        figure.ax.axhline(y=avg_beta, color='#808080', linestyle='--', linewidth=1.25, label='Average Beta')
        figure.add_text_box(text=f'Average Beta\n{round(avg_beta,2)}', x_coord=beta.index[120], y_coord=avg_beta, text_color='#808080', edge_color='none')

        figure.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%y'))
        figure.ax.set_ylim([min(beta)-0.2,max(beta)+0.2])
        figure.ax.legend(loc='lower center', fontsize=13, ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.18))
        figure.ax.yaxis.set_ticks_position('none')
        figure.ax.tick_params(axis='x', which='major', bottom=True, top=False, length=5, width=1.25)
        figure.end_plot(save_path=self.portfolio_instance.save_path, file_title='rolling_12m_beta')


    def corr_matrix(self):
        ''' computes the correlation matrix '''

        tickers = ['^VIX','^TNX','GC=F','BTC-USD']
        data = MarketData(tickers, '2020-01-01').get_other_prices('Adj Close')
        data['S&P'] = self.portfolio_instance.sp_data
        data = data.loc[self.portfolio_instance.performance_wealth['Strategy'].index]
        data['Strategy'] = self.portfolio_instance.performance_wealth['Strategy']
        data = data[['Strategy','S&P','^VIX','^TNX','GC=F','BTC-USD']]
        return data.corr().iloc[:-1,1:].round(2)


    def last_std(self):
        ''' computes the sharpe ratio for the last period '''

        last_period_perf = self.portfolio_instance.last_period_rescale(self.portfolio_instance.active_perf)
        last_period_std = last_period_perf.pct_change().std() * np.sqrt(252)
        return last_period_std


    def std_distribution(self):
        ''' simulates a sample of returns that could have been drawn in the investment universe '''

        annual_std = map(lambda x: (x.std()*np.sqrt(252)), self.portfolio_instance.random_returns)
        return list(annual_std)


    def std_distrib_plot(self):
        ''' plot the random sampling std distrib '''

        std_distrib = self.std_distribution()

        def std_prepare():
            last_period_std = self.last_std()
            last_period_bench = self.portfolio_instance.last_period_rescale(self.portfolio_instance.new_benchmark_perf)
            std_bench = last_period_bench.pct_change().std() * np.sqrt(252)
            last_period_no_reb = self.portfolio_instance.last_period_rescale(self.portfolio_instance.no_rebalance_perf)
            std_no_reb = last_period_no_reb.pct_change().std() * np.sqrt(252)
            return last_period_std, std_bench, std_no_reb

        last_period_std, std_bench, std_no_reb = std_prepare()
        percentile = round(stats.percentileofscore(std_distrib, self.last_period_std), 1)

        figure = GraphicsManager(title='Annualized volatility, strategy compared to random sample', max_xaxis_ticks=10, x_label='Annualized volatility', y_label='Density')

        sns.histplot(std_distrib, bins=30, stat='density', color='#2F5496', alpha=0.5)
        sns.kdeplot(std_distrib, color='#808080')
        figure.ax.axvline(x=self.last_period_std, color='black', linestyle='--', linewidth=2, label='Strategy')
        figure.add_text_box(text=f'Percentile\n{percentile}%', x_coord=self.last_period_std, y_coord=2)
        figure.ax.axvline(x=std_bench, color='#808080', linestyle='--', linewidth=2, label='Buy and hold')
        figure.ax.axvline(x=std_no_reb, color='#808080', linestyle='-.', linewidth=2, label='Strategy without EOM rebalance')
        if percentile < 85:
            plt.xlim([0, sorted(std_distrib)[-2]])

        figure.ax.tick_params(axis='both', colors='black')
        figure.ax.yaxis.set_ticks_position('none')
        figure.ax.legend(loc='lower center', fontsize=13, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.25))
        figure.end_plot(save_path=self.portfolio_instance.save_path, file_title='strategy_std_ditrib')
































from datetime import datetime
import yfinance as yf


from Setup import pd, np
from Setup.inputs import START_DATE, ASSETS_TICKER
from Data.Scrapping.prices_database import DataBaseQuery




### 1. Class related to data scrapping / query / loading
###############################################################################

class MarketData:
    ''' class related to data query / extraction '''

    def __init__(self, ticker, start_date):
        self.interval = '1d'
        self.ticker = ticker
        self.start_date = start_date
        self.db_instance = DataBaseQuery()


    def get_database_prices(self):
        ''' download ticker close prices from, yahoo finance '''

        prices = self.db_instance.data_query(self.ticker)
        prices = prices.iloc[prices.index >= self.start_date]
        prices.index = pd.to_datetime(prices.index)
        return prices


    def get_other_prices(self, price_type):
        ''' download ticker open prices from, yahoo finance '''

        prices = yf.download(self.ticker, start=self.start_date, interval=self.interval)[price_type]
        prices.index = pd.to_datetime(prices.index)
        return prices


    def normalize_data(self, data):
        ''' normalize the given data, useful for macro indicators '''

        return (data - data.mean()) / data.std()







### 1. Data setup
###############################################################################

__missing_data = ['ATVI']

ASSETS_TICKER_temp = [x for x in ASSETS_TICKER if x not in __missing_data]

ASSETS_INSTANCE = MarketData(ASSETS_TICKER_temp, START_DATE)
# ASSETS_PRICES = ASSETS_INSTANCE.get_database_prices()
ASSETS_PRICES = ASSETS_INSTANCE.get_other_prices('Adj Close')








'''
###################################################
fix ATVI missing
###################################################
'''

ASSETS_PRICES_fix = MarketData(__missing_data, START_DATE).get_database_prices()
ASSETS_PRICES[__missing_data] = ASSETS_PRICES_fix.loc[ASSETS_PRICES_fix.index >= ASSETS_PRICES.index[0]]


## defines the dates applicable to each stock (ex: if change in investment universe or M&A causing unlisting etc)
_assets_dates = {'start_date':[START_DATE for x in ASSETS_TICKER],
                  'end_date':[datetime.now().strftime('%Y-%m-%d') for x in ASSETS_TICKER]}
ASSETS_DATES = pd.DataFrame(data=_assets_dates, index=ASSETS_TICKER).T


# example to change an entry / out date for a stock; it will not be considered for strat / benchmark after (/before) this date
# ASSETS_DATES['ATVI']['end_date'] = '2023-10-21'
ASSETS_DATES['ATVI']['end_date'] = '2023-08-25'

# construct a date matrix that will be multiplied by the signals to remove the uneligible tickers by date
def __dates_matrix(matrix, ticker, dates_input):
    return (matrix.index >= dates_input[ticker]['start_date']) * (matrix.index <= dates_input[ticker]['end_date'])


DATES_MATRIX = pd.DataFrame(np.ones(ASSETS_PRICES.shape), index=ASSETS_PRICES.index, columns=ASSETS_PRICES.columns)
prices_fix = ASSETS_PRICES.copy(deep=True)


for ticker in DATES_MATRIX.columns:
    DATES_MATRIX[ticker] = __dates_matrix(DATES_MATRIX, ticker, ASSETS_DATES)
    mask = (prices_fix.index <= ASSETS_DATES[ticker]['start_date']) | (prices_fix.index >= ASSETS_DATES[ticker]['end_date'])
    prices_fix.loc[mask, ticker] = prices_fix[ticker].ffill()



prices_fix.dropna(how='any', axis=0, inplace=True)







### 2. Using S&P500 data
###############################################################################


# ASSETS_TICKER = pd.read_excel(r'C:/Users/const/OneDrive/Documents/Code/Python/Cresous_v2/Data/SP500_tickers.xlsx', index_col=0)['Symbol'].to_list()
# START_DATE = '2018-01-01'

# ASSETS_PRICES = pd.read_excel(r'C:/Users/const/OneDrive/Documents/Code/Python/Cresous_v2/Data/SP500_prices.xlsx', index_col=0)
# assets_returns = FinancialMetrics(ASSETS_PRICES).returns()


# ASSETS_INSTANCE = MarketData(ASSETS_TICKER, START_DATE)
# ASSETS_PRICES = ASSETS_INSTANCE.close_prices
# assets_returns = FinancialMetrics(ASSETS_PRICES).returns()
# ASSETS_PRICES.to_excel('Data/SP500_prices.xlsx')




# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:06:04 2024

@author: const
"""

import pandas as pd, numpy as np
from Setup.data_setup import MarketData, DATES_MATRIX
from Setup.financials import Strategy
from Setup.general import Plotter
from Strategies.nowcasting import Nowcaster




z = MarketData('FSLR', '2016-06-01').get_other_prices('Close')
z = pd.DataFrame(z)
sig = Nowcaster(z).trade_signal()




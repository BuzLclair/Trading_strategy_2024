from Strategies import pd, np, u_fin, u_gen, ASSETS_PRICES, ASSETS_TICKER, prices_fix




### 1. Benchmark (Equi-Weighting without rebalancing)
###############################################################################

benchmark = {}
benchmark['signal'] = pd.DataFrame(np.ones(prices_fix.shape), index=prices_fix.index, columns=ASSETS_TICKER).astype(bool)





benchmark['weights'] = u_fin.Weights(benchmark['signal'], frequency='monthly').equi_weighting()
benchmark['strategy'] = u_fin.Strategy(benchmark['weights'], prices_fix, rebalance=False)




if __name__ == '__main__':
    strategy_plot = u_gen.Plotter(benchmark['strategy'].cumulative_returns())
    strategy_plot.single_plot('Equi-weight benchmark return')






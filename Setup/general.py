import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore

from Setup import pd
from Setup.financials import VixAnalysis


### 1. Class related to graphic plotting
###############################################################################

class Plotter:
    ''' class related to plot, init with one curve, others can be added
        afterwards when calling multi-plotting methods '''

    def __init__(self, curve):
        self.curve = curve
        self.generic_setup()


    def generic_setup(self):
        ''' set up the generic type layout '''

        sns.set(rc = {"figure.figsize":(10,5), 'figure.dpi':600})
        sns.set_style({'grid.color':'lightgrey','grid.linestyle': ':',
                       'axes.facecolor': 'whitesmoke', 'axes.linewidth': 1,
                       'axes.edgecolor':'dimgrey'})


    def generic_end_commands(self):
        ''' run the last usual commands in a plot '''

        plt.gca().xaxis.grid(False)
        plt.gca().spines[['top','right']].set_color('none')
        plt.tight_layout()
        plt.show()


    def single_plot(self, title=None):
        ''' plot the curves given in init only '''

        sns.lineplot(self.curve, linewidth=1)
        plt.title(title, fontsize=15)
        self.generic_end_commands()


    def multi_plot(self, curves, title=None, legends=None):
        ''' plot the the additional curves given (curves var must be a list) '''

        plt.plot(self.curve, linewidth=1)
        for curve in curves:
            plt.plot(curve, linewidth=1)
        plt.title(title, fontsize=15)
        if legends != None:
            plt.legend(legends)
        self.generic_end_commands()



class Heatmaps:
    ''' class to plot heatmaps '''

    def __init__(self, returns_table):
        self.ret_table = returns_table
        self.cmap = sns.diverging_palette(10, 133, as_cmap=True)
        self.heatmap_setup()


    def heatmap_setup(self):
        ''' set up the generic type layout '''

        sns.set(rc = {"figure.figsize":(8,8), 'figure.dpi':600})
        sns.set_style({'grid.color':'white','axes.linewidth': 1,'figure.facecolor':'lightgray'})


    def generic_end_commands(self):
        ''' run the last usual commands in a plot '''

        plt.gca().xaxis.grid(False)
        plt.gca().spines[['top','right']].set_color('none')
        plt.tight_layout()
        plt.show()


    def plotter(self):
        ''' plot the heatmap '''

        sns.heatmap(self.ret_table, annot=True, linewidth=.3, cmap=self.cmap, vmin=-15, vmax=15,
                    center=0, linecolor='black', square=True, cbar=False, fmt='.1f')
        plt.title('Strategy monthly returns', fontsize=15)
        self.generic_end_commands()




class DistribPlot:
    ''' class to plot sharpe ratios distribution '''

    def __init__(self, distrib, strat_sharpe):
        self.distrib = distrib
        self.strat_sharpe = strat_sharpe
        self.percentile = round(percentileofscore(self.distrib, self.strat_sharpe), 1)
        self.size = len(distrib)
        self.plot_setup()


    def plot_setup(self):
        ''' set up the generic type layout '''

        sns.set(rc = {'figure.dpi':600})
        sns.set_style({'grid.color':'lightgrey','grid.linestyle': ':','axes.facecolor': 'whitesmoke',
                       'axes.linewidth': 1,'axes.edgecolor':'dimgrey'})


    def generic_end_commands(self):
        ''' run the last usual commands in a plot '''

        plt.gca().xaxis.grid(False)
        plt.gca().spines[['top','right']].set_color('none')
        plt.tight_layout()
        plt.show()


    def plotter(self):
        ''' plot the heatmap '''

        sns.displot(self.distrib, bins=20, kde=True, height=5, aspect=2)
        plt.axvline(x=self.strat_sharpe, color='red')
        plt.text(self.strat_sharpe * 1.02, 0.1*len(self.distrib), f'Percentile: {self.percentile}%', horizontalalignment='left', color='red')
        plt.title(f'Strategy sharpe ratio vs bootstraped distribution ({self.size} simulations)', fontsize=15)
        self.generic_end_commands()





### 2. VIX Plot
###############################################################################


class VixPlot(Plotter):
    ''' plots VIX related perf '''

    def __init__(self, strat_perf):
        Plotter.__init__(self, strat_perf)
        self.VIX_instance = VixAnalysis(strat_perf)


    def visual_returns(self):
        ''' plot the strategy returns clustered by VIX values '''

        perf = self.VIX_instance.perf
        sns.lineplot(data=perf, x=perf.index, y='Cumul returns', sort=False, color='gray', linewidth=1)
        resampled_data = perf.resample('3D').max()
        sns.scatterplot(data=resampled_data, x=resampled_data.index, y='Cumul returns', hue='Nowcaster', palette='Set1',
                        hue_order=['Contraction','Recovery','Slowdown','Expansion'])
        plt.title('Strategy Returns by VIX regime', fontsize=15)
        self.generic_end_commands()


    def visual_sharpe(self, benchmark):
        ''' plot the strategy sharpe ratios vs benchmark clustered by VIX values '''

        strat_sr = self.VIX_instance.sharpes
        strat_sr['Source'] = 'Strategy'
        benchmark_sr = VixAnalysis(benchmark).sharpes
        benchmark_sr['Source'] = 'Benchmark'
        sharpes = pd.concat([strat_sr, benchmark_sr], keys=['Strategy','Benchmark'])
        sharpes.reset_index(level=1, inplace=True)
        sharpes.columns = ['Nowcaster','Sharpe ratio','Source']
        sns.barplot(data=sharpes, x='Nowcaster', y='Sharpe ratio', hue='Source', palette=['royalblue','coral'])
        plt.title('Sharpe ratios by VIX regime', fontsize=15)
        self.generic_end_commands()





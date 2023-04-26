import pandas as pd
import numpy as np
import random
#thanks to https://github.com/rian-dolphin/Efficient-Frontier-Python

#because generating nf portfolios is time consuming, so we use the data pre-generated
pre_data = [{'mean_variance': [0.23152799775985275, 0.07936806139954913], 'weights': [0.01695096769401707, 0.4483097472545891, 0.0487046754417008, 0.46654578427556964, 0.019488825334123522], 'tickers': ['MSFT', 'AMGN', 'TGT', 'AAPL', 'JPM']}, 
            {'mean_variance': [0.1467002032864354, 0.04767134677156544], 'weights': [0.2338373851683184, 0.1813074695809701, 0.2012208531326655, 0.1635834947204308, 0.22005079739761513], 'tickers': ['WMT', 'MSFT', 'AMGN', 'TGT', 'AAPL']}, 
            {'mean_variance': [0.1779385926436864, 0.05406001156635057], 'weights': [0.33568096735680386, 0.001275069228845644, 0.07497494935079409, 0.29884532278011544, 0.2892236912834409], 'tickers': ['AAPL', 'MSFT', 'TGT', 'WMT', 'AMGN']}, 
            {'mean_variance': [0.22659481963151068, 0.08003555106133524], 'weights': [0.23384558658329915, 0.01635494380082396, 0.017025172633406773, 0.23142491106705396, 0.5013493859154161], 'tickers': ['MSFT', 'WMT', 'JPM', 'TGT', 'AAPL']}, 
            {'mean_variance': [0.12205988412509437, 0.04358499862730192], 'weights': [0.053980597463730205, 0.17006825183320023, 0.49535404042055975, 0.05906616687052091, 0.22153094341198887], 'tickers': ['MSFT', 'AAPL', 'WMT', 'TGT', 'AMGN']}, 
            {'mean_variance': [0.13422286212355145, 0.04500273121909828], 'weights': [0.19957397771609048, 0.17977252937070232, 0.15454813493145442, 0.11945234163991997, 0.3466530163418328], 'tickers': ['AAPL', 'MSFT', 'AMGN', 'TGT', 'WMT']}, 
            {'mean_variance': [0.23222745137668424, 0.07605535560412119], 'weights': [0.08911566659928309, 0.4902876135730243, 0.17327084155867242, 0.1798523974734913, 0.06747348079552896], 'tickers': ['MSFT', 'AAPL', 'TGT', 'AMGN', 'JPM']}, 
            {'mean_variance': [0.17128758122536297, 0.05218794564867525], 'weights': [0.2686657134519903, 0.12563714545303153, 0.2858295414468828, 0.30705815690663774, 0.012809442741457549], 'tickers': ['AMGN', 'TGT', 'WMT', 'AAPL', 'JPM']}, 
            {'mean_variance': [0.20554779027155856, 0.06575906122499511], 'weights': [0.3872148521867341, 0.04482331907798487, 0.3890938992862796, 0.08177930760501671, 0.09708862184398465], 'tickers': ['AAPL', 'JPM', 'AMGN', 'TGT', 'WMT']}, 
            {'mean_variance': [0.2124600591069549, 0.06767134586905502], 'weights': [0.45897146546849765, 0.12110479450403995, 0.13376103048352092, 0.16061446488745787, 0.12554824465648354], 'tickers': ['AAPL', 'AMGN', 'WMT', 'MSFT', 'TGT']}, 
            {'mean_variance': [0.17795404282996546, 0.05487311615657992], 'weights': [0.3367966451290694, 0.3453696431883094, 0.07585586388572542, 0.10554950488665715, 0.1364283429102386], 'tickers': ['WMT', 'AAPL', 'TGT', 'JPM', 'AMGN']}, 
            {'mean_variance': [0.11355344068076723, 0.042674954426447705], 'weights': [0.13208420553535632, 0.1176528554027348, 0.19626620170765055, 0.12835130530881333, 0.4256454320454449], 'tickers': ['TGT', 'MSFT', 'AMGN', 'AAPL', 'WMT']}, 
            {'mean_variance': [0.17753377754340438, 0.05414532834084024], 'weights': [0.3843162608368792, 0.14579297740534414, 0.04664297037266506, 0.06416955433176398, 0.35907823705334757], 'tickers': ['WMT', 'AMGN', 'JPM', 'TGT', 'AAPL']}, 
            {'mean_variance': [0.18991774338557857, 0.057243468362651394], 'weights': [0.2586639791810094, 0.3720083799587065, 0.22976141992807594, 0.10584781630731682, 0.03371840462489131], 'tickers': ['AMGN', 'AAPL', 'WMT', 'TGT', 'MSFT']}, 
            {'mean_variance': [0.26460737183943744, 0.09569074840971101], 'weights': [0.11382271420218804, 0.02233187889897309, 0.18130447892400708, 0.6449456046250851, 0.03759532334974665], 'tickers': ['MSFT', 'AMGN', 'TGT', 'AAPL', 'WMT']}
]
class Portfolio():
    @staticmethod
    def load_data():
        daily_returns = pd.read_csv('asset/daily_returns.csv', index_col=0)
        mus = (1+daily_returns.mean())**252 - 1
        cov = daily_returns.cov()*252
        return daily_returns, mus, cov

    @staticmethod
    def get_random_portfolios(n_assets=5, n_portfolios=1000):
        daily_returns, mus, cov = Portfolio.load_data()
        mean_variance_pairs = []
        np.random.seed(917101)

        for i in range(n_portfolios):
            assets = np.random.choice(list(daily_returns.columns), n_assets, replace=False)
            weights = np.random.rand(n_assets)
            weights = weights/sum(weights)
            portfolio_E_Variance = 0
            portfolio_E_Return = 0
            for i in range(len(assets)):
                portfolio_E_Return += weights[i] * mus.loc[assets[i]]
                for j in range(len(assets)):
                    portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]
            mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])

    @staticmethod
    def get_portfolios_from_efficient_froniter(n_assets=5, sample_num=50000):
        daily_returns, mus, cov = Portfolio.load_data()
        mean_variance_pairs = []
        weights_list=[]
        tickers_list=[]
        for i in range(sample_num):
            next_i = False
            while True:
                assets = np.random.choice(list(daily_returns.columns), n_assets, replace=False)
                weights = np.random.rand(n_assets)
                weights = weights/sum(weights)
                portfolio_E_Variance = 0
                portfolio_E_Return = 0
                for i in range(len(assets)):
                    portfolio_E_Return += weights[i] * mus.loc[assets[i]]
                    for j in range(len(assets)):
                        portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]

                for R,V in mean_variance_pairs:
                    if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                        next_i = True
                        break
                if next_i:
                    break

                mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])
                weights_list.append(weights)
                tickers_list.append(assets)
                break
        return mean_variance_pairs, weights_list, tickers_list

    @staticmethod
    def get_n_portfolios(n_assets=5, n_portfolios=15, use_pre=True):
        if use_pre:
            return pre_data

        mean_variance_pairs, weights_list, tickers_list = \
            Portfolio.get_portfolios_from_efficient_froniter(n_assets=n_assets)
        sample = random.sample(list(range(len(mean_variance_pairs))), n_portfolios)
        res = [{
            'mean_variance': mean_variance_pairs[index],
            'weights': weights_list[index].tolist(),
            'tickers': tickers_list[index].tolist()
        } for index in sample ]
        return res
        
from audioop import mul
import pandas as pd
import numpy as np
import random


class Portfolio():
    def __init__(self) -> None:
        daily_returns, mus, cov = self.load_data()
        self.daily_returns = daily_returns
        mus.drop(mus.index[0:2], inplace=True)
        cov.drop(cov.index[0:2], inplace=True)
        cov.drop(['AAPL', 'JPM'], axis=1, inplace=True)
        self.mus = mus
        self.cov = cov
        self._init_a_b_c()

    def _init_a_b_c(self):
        port_num = len(self.mus)
        M = self.mus.to_numpy(dtype=np.float64).reshape(-1, 1)
        O = np.ones((port_num, 1), dtype=np.float64)
        cov_matrix = self.cov.to_numpy(dtype=np.float64)
        k = np.dot(np.dot(M.T, np.linalg.inv(cov_matrix)), O)
        l = np.dot(np.dot(M.T, np.linalg.inv(cov_matrix)), M)
        m = np.dot(np.dot(O.T, np.linalg.inv(cov_matrix)), O)
        g = (l*np.dot(np.linalg.inv(cov_matrix), O) - k*np.dot(np.linalg.inv(cov_matrix), M)) / (l*m-k**2)
        h = (m*np.dot(np.linalg.inv(cov_matrix), M) - k*np.dot(np.linalg.inv(cov_matrix), O)) / (l*m-k**2)
        self.a = np.dot(np.dot(h.T, cov_matrix), h).flatten()[0]
        self.b = 2 * np.dot(np.dot(g.T, cov_matrix), h).flatten()[0]
        self.c = np.dot(np.dot(g.T, cov_matrix), g).flatten()[0]

    def load_data(self):
        daily_returns = pd.read_csv('asset/daily_returns.csv', index_col=0)
        mus = (1+daily_returns.mean())**252 - 1
        cov = daily_returns.cov()*252

        return daily_returns, mus, cov

    def get_mu_range(self):
        return self.mus.max(), self.mus.min()

    def get_variance_by_mu(self, mu):
        return self.a*(mu**2) + self.b*mu + self.c

    def get_n_portfolios(self, n_assets=5, n_portfolios=15, use_pre=True):
        mu_max, mu_min = self.get_mu_range()
        mus = np.arange(mu_min, mu_max, (mu_max-mu_min) / (n_portfolios-1))
        vars = [self.get_variance_by_mu(mu) for mu in mus]
        res = [{
            'mean_variance': [a,b],
        } for a,b in zip(mus,vars) ]
        return res
        
# a = Portfolio()

# mu1, mu2 = a.get_mu_range()
# print(mu1, mu2)
# print(a.mus)
# print(a.cov)
# print(a.get_sigma_by_mu(mu1), a.get_sigma_by_mu(mu2))
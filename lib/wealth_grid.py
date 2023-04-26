import pandas as pd
import math

class WealthGrid():
    @staticmethod
    def gen(w_0, I, portfolio, cost_v, t, n, bankrupt=20):
        res = []
        downward_param = -1
        mean_max, mean_min, variance_max, variance_min = WealthGrid.get_max_min_mean_variance(portfolio)
        w_min = WealthGrid.get_w_min(mean=mean_min, variance=variance_max, I=I, cost_v=cost_v, w_0=w_0, t=t, bankrupt=bankrupt)
        w_max = WealthGrid.get_w_max(mean=mean_max, variance=variance_min, I=I, t=t, w_0=w_0)
        # cal hat(W)
        for i in range(1, n+1):
            now = w_min * ((w_max / w_min)**((i-1)/(n-1)))
            if downward_param == -1 and now > w_0:
                downward_param = now
            res.append(now)
        # cal W
        downward_param = math.log(downward_param, math.e) - math.log(w_0, math.e)
        for i in range(len(res)):
            res[i] = round(res[i] * (math.e ** -downward_param), 3)
        return res

    @staticmethod
    def get_w_min(mean, variance, I, cost_v, t, w_0, bankrupt):
        res = []
        for i in range(t+1):
            now = w_0*math.exp(((mean - variance**2) / 2)*i-3*variance*(i**(0.5)))
            for j in range(i):
                infusion = 0 if j not in I else I[j]
                cost = 0 if j not in cost_v else cost_v[j][-1]
                now += (infusion - cost) * math.exp(((mean - variance**2) / 2)*(i-j)-3*variance*((i-j)**(0.5)))
            if now > bankrupt and len(res) == 0:
                res.append(now)
            elif now > bankrupt and res[-1] > now:
                res[-1] = now
        return res[-1]

    @staticmethod
    def get_w_max(mean, variance, I, t, w_0):
        res = w_0
        for i in range(t+1):
            now = w_0*math.exp(((mean - variance**2) / 2)*i+3*variance*(i**(0.5)))
            for j in range(i):
                infusion = 0 if j not in I else I[j]
                now += (infusion) * math.exp(((mean - variance**2) / 2)*(i-j)+3*variance*((i-j)**(0.5)))
            if res < now:
                res = now
        return res

    @staticmethod    
    def get_max_min_mean_variance(portfolio):
        df = pd.json_normalize(portfolio)
        df['mean'] = df['mean_variance'].map(lambda x:x[0])
        df['variance'] = df['mean_variance'].map(lambda x:x[1])
        df = df[['mean', 'variance']]
        return df['mean'].max(), df['mean'].min(), df['variance'].max(), df['variance'].min() 
        

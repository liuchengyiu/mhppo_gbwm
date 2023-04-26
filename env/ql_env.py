from scipy import stats
import numpy as np

class QL_ENV():
    def __init__(self, util_v, cost_v, T, w_grid, w0_index, port):
        self.util_v = util_v
        self.cost_v = cost_v
        self.T = T
        self.t = 1
        self.w_indics = np.arange(0, len(w_grid))
        self.w_grid = np.array(w_grid)
        self.w0_index = w0_index
        self.portfolios = port

    def reset(self):
        self.t = 0
        return self.t, self.w0_index

    def step(self, w_i, I, action, h):
        cost, mu, sig = action[0], *self.portfolios[action[1]]
        remain = 1 if (w_i+I-cost) < 1 else (w_i+I-cost)
        p1 = stats.norm.pdf((np.log(self.w_grid /(remain)) -(mu -0.5* sig)*h)/(np.sqrt(sig)*np.sqrt(h)))
        # print(p1)
        # print(p1.sum())
        try:
            p1_ = p1
            p1 = p1/ p1.sum()
            idx = np.random.choice(self.w_indics, p=p1)
        except Exception as e:
            print(w_i+I-cost)
            print(e)
            print(p1_)
            print(p1.sum())
            exit(1)

        k = self.cost_v[self.t].index(int(cost))
        reward = self.util_v[self.t][k]
        self.t += 1
        if self.t == self.T+1:
            done = True
        else:
            done = False
        return idx, reward, self.t, done


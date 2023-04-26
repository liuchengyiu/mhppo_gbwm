import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dead_prob = pd.read_csv('./asset/morality_prob.csv')


class WhiteNoise():
    def __init__(self, cost_v, T):
        self.cost_v = cost_v
        self.T = T
        self._init_white_noise()

    def _init_white_noise(self):
        white = np.zeros((self.T+3, 3), dtype=np.float32)
        for i in range(self.T+2):
            if i not in self.cost_v:
                white[i, :] = np.array([1.,0,1.])
            else:
                cost = np.array(self.cost_v[i])
                min_, max_ = cost.min(), cost.max()+3
                var = np.std(cost) 
                white[i,:] = np.array([var, min_, max_])
        self.white = white

    def add_white_noise(self, s):
        t = int(s[1])
        var, left, right = self.white[t, :]
        
        s[2:] = np.clip(np.random.normal(s[2:], var), left, right)
        return s
    
    def white_decay(self, decay):
        self.white[:, 0] = self.white[:, 0]*decay
        # sleep(1)


class PDQN_ENV():
    def __init__(self, util_v, cost_v, T, w0, port, infusion):
        self.mu_bound = port.get_mu_range()
        self.util_v = util_v
        self.cost_v = cost_v
        self.T = T
        self.w0 = w0
        self._init_util_cost_v()
        self.port = port
        self._init_infusion(infusion)
        self.state_dim = 2 + len(self.cost_np[0][0])
        self.action_dim = len(self.cost_np[0][0])

    def _init_infusion(self, infusion):
        self.infusion = {}
        for i in range(self.T+2):
            self.infusion[i] = 0 if i not in infusion else infusion[i]

    def _init_util_cost_v(self):
        max_len = 0
        self.cost_np = []
        self.util_np = []
        for item in self.cost_v:
            item = self.cost_v[item]
            max_len = len(item) if len(item) > max_len else max_len
        self.c_index = np.arange(0,max_len,1)

        for i in range(self.T+2):
            c = np.zeros((1, max_len), dtype=np.float32)
            u = np.zeros((1, max_len), dtype=np.float32)
            if i in self.cost_v:
                c[:, :len(self.cost_v[i])] = self.cost_v[i]
                u[:, :len(self.util_v[i])] = self.util_v[i]
            self.cost_np.append(c)
            self.util_np.append(u)

    def _get_state(self, w_i):
        s = np.zeros((1, 2 + len(self.cost_np[self.t][0])), dtype=np.float32)
        s[0][0] = w_i + self.infusion[self.t]
        s[0][1] = self.t
        s[:,2:] = self.cost_np[self.t][0]
        return s.flatten()

    def reset(self):
        self.t = 0
        return self._get_state(self.w0)

    def step(self, s, action, h=1):
        w_i = s[0]
        mu = action[1]*(self.mu_bound[0] - self.mu_bound[1]) + self.mu_bound[1]
        variance = self.port.get_variance_by_mu(mu)
        cost_index = int(action[0])
        if self.cost_np[self.t][0][cost_index] <= w_i:
            cost = self.cost_np[self.t][0][cost_index]
            reward = self.util_np[self.t][0][cost_index]
        else:
            cost = 0
            reward = 0
        w_i = w_i - cost
        z = np.random.normal(0,1)
        w_next = w_i * np.exp((mu - 0.5 * variance)*h + np.sqrt(variance)*np.sqrt(h)*z)
        self.t += 1
        if self.t == self.T+1:
            done = True
        else:
            done = False
        return self._get_state(w_next+self.infusion[self.t]), reward, done


class Tools:
    @staticmethod
    def c_u_len(c_u):
        cu_len = 0
        postpone_goals = []
        for index, item in enumerate(c_u):
            postpone_goals = [ item for item in postpone_goals if item['t'] + item['postpone'] >= index]
            cu_list = []
            for it in item:
                cu_list.append(it['c_u'])
            for it in postpone_goals:
                cu_list.append(it['c_u'])
            cu_now = len(Tools._get_combine(cu_list))
            if cu_len < cu_now:
                cu_len = cu_now
            for it in item:
                if it['postpone'] != 0:
                    postpone_goals.append(it)
        return cu_len

    @staticmethod
    def _get_combine(cu_list):
        def sort_by_cost(arr):
            return arr[0]
        def sort_by_cost_v2(arr):
            return arr[0][0]
        if len(cu_list) == 0:
            return []
        cu_list = [sorted(it, key=sort_by_cost) for it in cu_list]
        for index in range(len(cu_list)):
            cu_list[index] = [[i]+it for i, it in enumerate(cu_list[index])]
        y_con_goals = []
        y_strategy = []
        for it in itertools.product(*cu_list):
            np_arr = np.array(it)
            stra, c_g = np_arr[:,:1],np_arr[:,1:]
            c_g = np.sum(c_g, axis=0)
            y_strategy.append(stra.flatten().tolist())
            y_con_goals.append(c_g.tolist())
        s_c_u = [[a, b] for a, b in zip(y_con_goals, y_strategy)]
        s_c_u.sort(key=sort_by_cost_v2)
        tmp = []
        for item in s_c_u:
            if len(tmp) == 0:
                tmp.append(item)
                continue
            if tmp[-1][0][1] >= item[0][1]:
                continue
            if tmp[-1][0][0] == item[0][0]:
                tmp.pop()
            tmp.append(item)
        return tmp
    
    @staticmethod
    def is_dead(age, sex):
        mortal = dead_prob.iloc[age]['prob_man'] if sex == 0 else dead_prob.iloc[age]['prob_woman']
        if np.random.uniform() < mortal:
            return True
        return False
    
    @staticmethod
    def compute_cu_param(c_u):
        def utility_func(x, k1, k2, a):
            return k1*(x**(1-a) / (1-a)) + k2
        x = np.array([ d[0] for item in c_u for it in item for d in it['c_u'] if d[0] != 0])
        y = np.array([ d[1] for item in c_u for it in item for d in it['c_u'] if d[0] != 0])
        popt1, _ = curve_fit(utility_func, x, y, maxfev = 500000, bounds=(0, [10000., 10000., 1.]))

        return popt1
    
class PDQNEnvV2():
    def __init__(self, c_u, T, w0, port, infusion, morality=None, mandatory=False, tex=None, auto_infusion=None):
        self.mandatory = mandatory
        self.tex = tex
        self.auto_infusion = auto_infusion
        self.cu_param = Tools.compute_cu_param(c_u)
        if morality is not None:
            self.age = morality['age']
            self.sex = morality['sex']
        self.mu_bound = port.get_mu_range()
        self.c_u = c_u
        self.T = T
        if len(c_u) < T+2:
            c_u += [[]]*(T+2-len(c_u))
        self.w0 = w0
        self.max_cu_len = Tools.c_u_len(self.c_u)
        self.port = port
        self._init_infusion(infusion)
        self.state_dim = 2 + self.max_cu_len
        self.action_dim = self.max_cu_len
        self.allow = True

    def _get_auto_infusion_reward(self, infusion):
        if infusion < 1e-6:
            return 0
        a = infusion ** (1-self.auto_infusion['p']) / (1-self.auto_infusion['p'])
        return - (self.cu_param[0] * a + self.cu_param[1]) 

    def _init_t_acton_np(self):
        now = self.c_u[self.t] + self.postpone_goals
        cu_list = [ item['c_u'] for item in now]
        c_u_now = Tools._get_combine(cu_list)
        c = [ item[0][0] for item in c_u_now ]
        self.c_u_now = c_u_now
        c += [0]*(self.max_cu_len-len(c))
        return [c]

    def _init_infusion(self, infusion):
        self.infusion = {}
        for i in range(self.T+2):
            self.infusion[i] = 0 if i not in infusion else infusion[i]

    def _get_state(self, w_i):
        s = np.zeros((1, 2 + self.max_cu_len), dtype=np.float32)
        s[0][0] = w_i + self.infusion[self.t]
        s[0][1] = self.t
        s[:,2:] = self._init_t_acton_np()
        return s.flatten()

    def reset(self):
        self.t = 0
        self.postpone_goals = []
        self.allow = True
        return self._get_state(self.w0)

    def postpone_goals_step(self, strategy):
        now = self.c_u[self.t-1] + self.postpone_goals
        postpone = []
        for index, item in enumerate(now):
            if item['t'] + item['postpone'] < self.t:
                if self.mandatory and item['mandatory']:
                    if strategy is None or strategy[index] == 0:
                        self.allow = False
                        return
                continue
            if strategy is not None:
                if strategy[index] != 0:
                    continue
            postpone.append(item)
        self.postpone_goals = postpone

    def step(self, s, action, h=1):
        w_i = s[0]
        mu = action[1][0]*(self.mu_bound[0] - self.mu_bound[1]) + self.mu_bound[1]
        variance = self.port.get_variance_by_mu(mu)
        cost_index = int(action[0])

        if self.auto_infusion is not None:
            infusion = action[1][1] * self.auto_infusion['max']
            # print(infusion)
            w_i += infusion

        if cost_index < len(self.c_u_now) and self.c_u_now[cost_index][0][0] <= w_i:
            cost = self.c_u_now[cost_index][0][0]
            reward = self.c_u_now[cost_index][0][1]
            strategy = self.c_u_now[cost_index][1]
        else:
            cost = 0
            reward = 0
            strategy = None
        w_i = w_i - cost
        z = np.random.normal(0,1)
        w_next = w_i * np.exp((mu - 0.5 * variance)*h + np.sqrt(variance)*np.sqrt(h)*z)
        self.t += 1

        if self.auto_infusion is not None:
            r = self._get_auto_infusion_reward(infusion)
            reward += r
            # print(r)

        if self.t == self.T+1:
            done = True
        else:
            done = False
        if hasattr(self, 'sex'):
            if not done:
                done = Tools.is_dead(age=self.age+self.t, sex=self.sex)
                if done:
                    reward -= 30
        if self.mandatory:
            if not self.allow:
                reward = -1000000
                done = True
        # if self.tex != None:
        #     if w_next > w_i:
        #         w_next = (w_next-w_i) *0.75

        self.postpone_goals_step(strategy)
        return self._get_state(w_next+self.infusion[self.t]), reward, done

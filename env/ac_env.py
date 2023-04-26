from os import access
from time import sleep
import numpy as np


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


class ACEnv():
    def __init__(self, util_v, cost_v, T, w0, port, infusion):
        self.util_v = util_v
        self.cost_v = cost_v
        self.T = T
        self.w0 = w0
        self._init_util_cost_v()
        self.port = port
        self._init_infusion(infusion)
        self.state_dim = 2 + len(self.cost_np[0][0])
        self.action_dim = len(port) * len(self.cost_np[0][0])
        self.action_dim1 = len(port)
        self.action_dim2 = len(self.cost_np[0][0])

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
        # action = int(action)
        # print(action)
        # l = action[0]
        # k = action[1]
        l = int(action / len(self.cost_np[0][0]))
        k = action % len(self.cost_np[0][0])
        mu, variance = self.port[l]
        if self.cost_np[self.t][0][k] <= w_i:
            cost = self.cost_np[self.t][0][k]
            reward = self.util_np[self.t][0][k]
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
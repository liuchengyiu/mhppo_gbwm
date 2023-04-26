import numpy as np
import pandas as pd
import itertools as it

class QLearningTable:
    def __init__(self, wealth_grid, portfolios, cost_v, util_v, T, learning_rate=1, reward_decay=1, e_greedy=0.9):
        self.portfolios = np.arange(0, len(portfolios), step=1)
        self.portfolios = np.reshape(self.portfolios, (len(portfolios),1))
        self.cost_v = cost_v
        self.util_v = util_v
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.T = T
        self.wealth_grid = wealth_grid
        self._build_action()
        self._init_Q_table()

    def _build_action(self):
        self.actions = []
        for t in range(self.T+1):
            cost_v = self.cost_v[t]
            df = np.append(np.full((len(self.portfolios), 1), cost_v[0]), self.portfolios, axis=1)
            self.actions.append(df)
            for cost in cost_v[1:]:
                df = np.append(np.full((len(self.portfolios), 1), cost), self.portfolios, axis=1)
                self.actions[-1] = np.append(self.actions[-1], df, axis=0)
                
    def _init_Q_table(self):
        self.Q = []
        for t in range(self.T+1):
            self.Q.append(np.zeros((len(self.wealth_grid), len(self.actions[t]))))
        # action_num = len(np.where(self.actions[t][:,0]<24)[0])
        # print(self.actions[t][:action_num])
        # self.Q[t][0, 20] = 2.0
        # q_state_action = pd.DataFrame(self.Q[t][0, :action_num])
        # print(np.random.choice(q_state_action[q_state_action[0] == np.max(q_state_action)[0]].index))

    def choose_action(self, w_index, infusion,t):
        # action selection
        w = self.wealth_grid[w_index] + infusion
        action_num = len(np.where(self.actions[t][:,0]<w)[0])
        if np.random.uniform() < self.epsilon:
            q_state_action = pd.DataFrame(self.Q[t][w_index, :action_num])
            # choose best action
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(q_state_action[q_state_action[0] == np.max(q_state_action)[0]].index)
        else:
            # choose random action
            action = np.random.randint(0, action_num)
        return action, self.actions[t][action]

    def learn(self, w_index, t, a, r, w_index_, t_):
        q_predict = self.Q[t][w_index][a]
        if t_ != self.T+1:
            q_target = r + self.gamma * self.Q[t_][w_index_,:].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.Q[t][w_index][a] += self.lr * (q_target - q_predict)  # update

    def get_best_strategy(self, w0_index):
        def get_action_detail(t, w_index):
            q_state_action =  pd.DataFrame(self.Q[t][w_index,:])
            if np.max(q_state_action)[0] == 0:
                return (-1,-1)
            action = np.random.choice(q_state_action[q_state_action[0] == np.max(q_state_action)[0]].index)
            detail = self.actions[t][action]
            k = self.cost_v[t].index(int(detail[0]))
            return int(detail[1]), k 
        strategy_g = {index: {} for index in range(len(self.wealth_grid))}
        for t in range(self.T+1):
            print(self.Q[t][w0_index,:])
            if t == 0:
                strategy_g[w0_index][t] = get_action_detail(t, w0_index)
                continue
            for index in range(len(self.wealth_grid)):
                strategy_g[index][t] = get_action_detail(t, index)
        return strategy_g

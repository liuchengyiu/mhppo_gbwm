from lib.wealth_grid import WealthGrid
from lib.portfolio_concreet import Portfolio
from lib.load import LoadData
import pandas as pd
import numpy as np
from agent.qlearningtable import QLearningTable
from env.ql_env import QL_ENV
from tqdm import trange
w_0 = 100
infusion = {0: 10, 1: 10}
t = 21
grid_points = 1000
H = 1
port = Portfolio().get_n_portfolios()
data, k_strategy, cost_v, util_v = LoadData.load("./asset/new_user_goal.yml")
wealth_grid = WealthGrid.gen(w_0, infusion, port, cost_v, t, grid_points)
cost_v = [cost_v[i] if i in cost_v else [0] for i in range(t+3)]
util_v = [util_v[i] if i in util_v else [0] for i in range(t+3)]
infusion = [infusion[i] if i in infusion else 0 for i in range(t+3)]
df = np.array(pd.DataFrame(port)['mean_variance'].to_list())
RL = QLearningTable(wealth_grid, df, cost_v, util_v, t)
env = QL_ENV(util_v, cost_v, t, wealth_grid, wealth_grid.index(w_0), df)
def update():
    for episode in trange(10000):
        # initial observation
        t, w_index = env.reset() 
        if episode < 50000:
            RL.epsilon = 0.5
        elif episode > 80000:
            RL.epsilon = 0.9
        else:
            RL.epsilon = 0.7
        while True:
            # fresh env

            # RL choose action based on observation
            action, action_detail = RL.choose_action(w_index, infusion[t], t)

            # RL take action and get next observation and reward
            w_index_, reward, t_, done = env.step(w_i = wealth_grid[w_index], I=infusion[t], action=action_detail, h=H)

            # RL learn from this transition
            RL.learn(w_index, t, action, reward, w_index_, t_)

            # swap observation
            t, w_index = t_, w_index_
            # break while loop when end of this episode
            if done:
                break

update()
def predict():
    reg = 0
    ep = 1000
    sigle_re = []
    for episode in trange(ep):
        re = 0
        t, w_index = env.reset() 
        while True:
            # fresh env

            # RL choose action based on observation
            action, action_detail = RL.choose_action(w_index, infusion[t], t)

            # RL take action and get next observation and reward
            w_index_, reward, t_, done = env.step(w_i = wealth_grid[w_index], I=infusion[t], action=action_detail, h=H)

            # RL learn from this transition
            # RL.learn(w_index, t, action, reward, w_index_, t_)
            reg += reward
            re += reward
            # swap observation
            t, w_index = t_, w_index_
            # break while loop when end of this episode
            if done:
                sigle_re.append(re)
                break
    df = pd.DataFrame(sigle_re)
    df.to_csv("q_table.csv")
    print(reg / ep)
predict()
# print(RL.get_best_strategy(env.w0_index))
# print(env.w0_index)
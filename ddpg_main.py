from lib.load import LoadData
from env.ddbg_env import DDPG_ENV, WhiteNoise
from agent.ddpg import DDPG
from lib.portfolio_concreet import Portfolio
import time
from tqdm import trange
import torch
import numpy as np

infusion = {0: 10, 1: 10}
MEMORY_CAPACITY = 3000
EPISODES = 10000
REPLACEMENT = [
    dict(name='soft', tau=0.005),
    dict(name='hard', rep_iter=600)
][0]
T = 7
data, k_strategy, cost_v, util_v = LoadData.load("./asset/user_goal.yml")
port = Portfolio()
mu_bound = port.get_mu_range()
env = DDPG_ENV(util_v, cost_v, T, 100, port, infusion, mu_bound)
s_dim = 2 + len(env.cost_np[0][0])
a_dim = 1 + len(env.cost_np[0][0])
ddpg = DDPG(action_dim=a_dim, state_dim=s_dim, mu_bound=mu_bound, replacement=REPLACEMENT)
t1 = time.time()
H=1
noise = WhiteNoise(cost_v, T)
def train():
    VAR1 = 1
    VAR2 = 1
    for _ in trange(EPISODES):
        s = env.reset()
        ep_r = 0
        while True:
            # add explorative noise to action
            a = ddpg.choose_action(s)
            # a = np.array(a) 
            # a[0] = np.clip(np.random.normal(a[0], VAR1), mu_bound[1], mu_bound[0])
            # a[1:] = np.clip(np.random.normal(a[1:], VAR2), 0, 1)
            a = torch.FloatTensor(a)
            s_, r, done = env.step(s[0], a, h=H)
            ep_r += r
            # s_ = noise.add_white_noise(s_)
            # ddpg.store_transition(s, a, r, s_) # store the transition to memory
            s = s_
            if done:
                ddpg.store_transition(s, a, ep_r, s_)
                if ddpg.pointer > MEMORY_CAPACITY:
                    # VAR1 *= 0.95
                    # VAR2 *= 0.95
                    # noise.white_decay(0.98)
                    ddpg.learn()
                break
            else:
                ddpg.store_transition(s, a, 0, s_)
                if ddpg.pointer > MEMORY_CAPACITY:
                    ddpg.learn()

def predict():
    ddpg.load("./models")
    ep_r = 0 
    for i in trange(10000):
        s = env.reset()
        while True:
            # add explorative noise to action
            a = ddpg.actor_target(torch.unsqueeze(torch.FloatTensor(s), dim=0))[0].detach()
            s_, r, done = env.step(s[0], a, h=H)
            # ddpg.store_transition(s, a, r, s_) # store the transition to memory
            
            # if ddpg.pointer > MEMORY_CAPACITY and ddpg.pointer % 100 == 0:
            #     ddpg.learn()
            s = s_
            ep_r += r
            if done:
                break 
    print('Running time:{} R: {}'.format(time.time() - t1, ep_r/10000))
# ddpg.load("./models")
train()
ddpg.save("./models")
predict()
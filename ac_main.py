# from agent.ac_before import Actor, Critic, Policy
from agent.ac import Actor, Critic
from lib.portfolio_concreet import Portfolio
from lib.load import LoadData
from env.ac_env import ACEnv
import torch
import numpy as np
from time import sleep
infusion = {0: 10, 1: 10}
MEMORY_CAPACITY = 3000
EPISODES = 10000
REPLACEMENT = [
    dict(name='soft', tau=0.005),
    dict(name='hard', rep_iter=600)
][0]
T = 7
H=1
data, k_strategy, cost_v, util_v = LoadData.load("./asset/user_goal.yml")
port = Portfolio().get_n_portfolios()
port = [item['mean_variance'] for item in port]
env = ACEnv(util_v, cost_v, T, 100, port, infusion)
actor=Actor(env)
critic=Critic(env.state_dim)


EPISODE = 30000  # Episode limitation
TEST = 1000  # The number of experiment test every 100 episode
STEP = 5000

def main():
    # actor = Actor(env)
    # critic = Critic(env)

    for episode in range(EPISODE):
        state = env.reset()

        while 1:
            action = actor.choose_action(state)
            next_state, reward, done = env.step(state, action, H)
            td_error = critic.learn(state, reward, next_state)  
            actor.learn(state, action, td_error)
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 1000 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                while 1:
                    action =  actor.choose_action(state, max=True)  # direct action for test
                    # print(action)
                    state, reward, done = env.step(state, action, H)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)

def lab_td():
    model = Policy(env.state_dim, env.action_dim)

    epis = 30000
    for episode in range(epis):
        state = env.reset()
        while 1:
            action = model.choose_action(state, store=True)
            next_state, reward, done = env.step(state, action, H)
            model.rewards.append(reward)
            state = next_state
            if done:
                break
        model.learn()

        if episode % TEST == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                while 1:
                    action = model.choose_action(state, max=True)  # direct action for test
                    state, reward, done = env.step(state, action, H)
                    total_reward += reward
                    print(reward, env.t, state[0])
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        
main()
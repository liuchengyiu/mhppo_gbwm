import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from lib.normalization import Normalization, RewardScaling
from lib.replaybuffer import ReplayBuffer,ReplayBufferV2, ReplayBufferV3
from agent.td3 import TD3
from lib.portfolio_concreet import Portfolio
from lib.load import LoadData2
from env.hppo_env import HPPOEnvV2
import torch
import numpy as np
from time import sleep
import datetime

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

def evaluate_policy(args, env, agent):
    times = 100
    evaluate_reward = 0
    mu = 0
    infu=0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            act, act_param, all_actions, all_action_parameters = agent.choose_action(s, max=True)
            param = (act_param+1)/2
            s_, r, done = env.step(s, (act, param))
            episode_reward += r
            mu += param[0]
            infu += param[1]
            s = s_
        evaluate_reward += episode_reward
    print("mu:", mu/times)
    print('infu:', infu)
    return evaluate_reward / times


def main(args, env_name, number, seed):
    infusion = {0: 10, 1: 10}
    T = 21
    c_u = LoadData2.load("./asset/new_user_goal.yml")
    port = Portfolio()
    env = HPPOEnvV2(c_u, T, 100, port, infusion, 
            # morality={'age':40, 'sex':0}, 
            mandatory=False,
            auto_infusion={'max': 30,'p': 0.01,}
    )
    env_evaluate = HPPOEnvV2(c_u, T, 100, port, infusion, 
            # morality={'age':40, 'sex':0}, 
            mandatory=False,
            auto_infusion={'max': 30,'p': 0.01,}
    )
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.state_dim
    args.action_dim = env.action_dim
    args.max_action = 1
    args.max_episode_steps = T+3  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    agent = TD3(
        action_dim=env.action_dim,
        state_dim=env.state_dim,
        param_dim=env.action_dim*2,
    )
    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))
    reward_scaling = RewardScaling(shape=1, gamma=0.99)
    reward_scaling.reset()
    episode_steps = 0
    evaluate_num = 0
    for episode_steps in range(1000000):
        s = env.reset()
        episode_steps += 1
        act, act_param, all_actions, all_action_parameters = agent.choose_action(s)
        done = False
        agent.start_episode()

        while not done:
            s_, r, done = env.step(s, (act, (act_param+1)/2))
            next_act, next_act_param, next_all_actions, next_all_action_parameters = agent.choose_action(s_)
            # r_ = reward_scaling(r)
            agent.step(s, (all_actions, all_action_parameters), r, s_, done,
                       optimise=True)
            s = s_
            act, act_param, all_actions, all_action_parameters = next_act, next_act_param, next_all_actions, next_all_action_parameters
        agent.end_episode()
        if episode_steps % 1000 == 0:
            evaluate_num += 1
            evaluate_reward = evaluate_policy(args, env_evaluate, agent)
            print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=9e4, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['CartPole-v1', 'LunarLander-v2']
    env_index = 1
    main(args, env_name=env_name[env_index], number=1, seed=0)
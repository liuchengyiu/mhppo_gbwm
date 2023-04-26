import torch
import numpy as np
from agent.p_dqn import PDQNAgent
from lib.portfolio_concreet import Portfolio
from lib.load import LoadData2
from env.p_dqn_env import PDQNEnvV2
import torch
import numpy as np
from time import sleep
from tqdm import trange

def evaluate_policy(env, agent):
    times = 1000
    evaluate_reward = 0
    mu = 0
    infu = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, action_parameters, all_action_parameters = agent.act(s, max=True)  # We use the deterministic policy during the evaluating
            s_, r, done = env.step(s, (action, action_parameters))
            episode_reward += r
            mu += action_parameters[0]
            # infu += action_parameters[1]
            s = s_
        evaluate_reward += episode_reward
    print("mu:", mu/times)
    # print('infu:', infu)
    return evaluate_reward / times


def main():
    infusion = {0: 10, 1: 10}
    T = 21
    c_u = LoadData2.load("./asset/new_user_goal.yml")
    port = Portfolio()
    env = PDQNEnvV2(c_u, T, 100, port, infusion, morality={'age':40, 'sex':0}, mandatory=False,
            # auto_infusion={'max': 30,'p': 0.01,}
    )
    env_evaluate = PDQNEnvV2(c_u, T, 100, port, infusion, morality={'age':40, 'sex':0}, mandatory=False, 
            # auto_infusion={'max': 30, 'p': 0.01,}
    )
    # Set random seed
    np.random.seed(1)
    torch.manual_seed(1)

    print("state_dim={}".format(env.state_dim))
    print("action_dim={}".format(env.action_dim))
    actor_type = 'dqn'
    print(actor_type)
    agent = PDQNAgent(env.state_dim, 
                        env.action_dim, 
                        # double_space=True, 
                        actor_type=actor_type
                        )

    # Build a tensorboard

    for i in range(100000):
        s = env.reset()
        action, action_parameters, all_action_parameters = agent.act(s) 
        episode_steps = 0
        done = False
        reward = 0
        while not done:
            episode_steps += 1
            s_, r, done = env.step(s, (action, action_parameters))
            next_act, next_act_param, next_all_action_parameters = agent.act(s_)
            agent.step(s, (action, all_action_parameters), r, s_,
                    terminal = done)
            reward += r
            action, action_parameters, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            s = s_
        agent.end_episode()
        if i % 1000 == 0:
            print("epo: {} reward:{}", i, evaluate_policy(env_evaluate, agent))
if __name__ == '__main__':
    main()
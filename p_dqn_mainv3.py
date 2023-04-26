import torch
import numpy as np
from agent.pdqn import QActorDualing, PDQNAgent
from lib.portfolio_concreet import Portfolio
from lib.load import LoadData2
from env.p_dqn_env import PDQNEnvV2
import torch
import numpy as np
from time import sleep
from tqdm import trange
import click
import ast

class ClickPythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(e)
            raise click.BadParameter(value)

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
            infu += action_parameters[1]
            s = s_
        evaluate_reward += episode_reward
    print("mu:", mu/times)
    print('infu:', infu)
    return evaluate_reward / times


@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=20000, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500??
@click.option('--use-ornstein-noise', default=True,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=100000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.0001, help="Actor network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)  # 0.00001
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--layers', default='[128,]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=True, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="PDDQN", help="Prefix of output files", type=str)
def main(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title):
    infusion = {0: 10, 1: 10}
    T = 21
    c_u = LoadData2.load("./asset/new_user_goal.yml")
    port = Portfolio()
    env = PDQNEnvV2(c_u, T, 100, port, infusion, morality={'age':40, 'sex':0}, mandatory=False,
            auto_infusion={'max': 30,'p': 0.01,}
    )
    env_evaluate = PDQNEnvV2(c_u, T, 100, port, infusion, morality={'age':40, 'sex':0}, mandatory=False, 
            auto_infusion={'max': 30, 'p': 0.01,}
    )
    # Set random seed
    np.random.seed(1)
    torch.manual_seed(1)

    print("state_dim={}".format(env.state_dim))
    print("action_dim={}".format(env.action_dim))
    actor_type = 'dqn'
    print(actor_type)
    agent = PDQNAgent(
                        env.state_dim, env.action_dim,
                        actor_class=QActorDualing,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,
                       learning_rate_actor_param=learning_rate_actor_param,
                       epsilon_steps=epsilon_steps,
                       gamma=gamma,
                       tau_actor=tau_actor,
                       tau_actor_param=tau_actor_param,
                       clip_grad=clip_grad,
                       indexed=indexed,
                       weighted=weighted,
                       average=average,
                       random_weighted=random_weighted,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': [512,521,256,256],
                                     'action_input_layer': action_input_layer,},
                       actor_param_kwargs={'hidden_layers': [256],
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,},
                       zero_index_gradients=zero_index_gradients,
                       seed=seed
    )

    # Build a tensorboard
    import tqdm
    for i in tqdm.trange(100000):
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
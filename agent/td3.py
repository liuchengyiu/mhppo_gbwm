import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from lib.memory import Memory
from torch.autograd import Variable
from torch.distributions import Categorical

def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)

def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def normal_init(layer, init_std):
    nn.init.normal_(layer.weight.data, std=init_std)
    nn.init.zeros_(layer.bias.data)

def xavier_init(layer):
    tanh_gain = nn.init.calculate_gain('tanh')
    nn.init.xavier_normal_(layer.weight, tanh_gain)
    nn.init.zeros_(layer.bias)

def kaiming_normal(layer):
    nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
    nn.init.zeros_(layer.bias.data)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(Actor, self).__init__()
        self.hidden = (1024, 512, 512, 512)
        self.layers = nn.ModuleList()
        last_dim = state_dim
        for i in range(len(self.hidden)):
            self.layers.append(nn.Linear(last_dim, self.hidden[i]))
            xavier_init(self.layers[i])
            last_dim = self.hidden[i]
        self.action_layer = nn.Linear(last_dim, action_dim)
        xavier_init(self.action_layer)
    
    def forward(self, s):
        for i in range(len(self.layers)):
            s = F.relu(self.layers[i](s))
        return self.action_layer(s)

class ActorParam(nn.Module):
    def __init__(self, state_dim, param_dim) -> None:
        super(ActorParam, self).__init__()
        self.hidden = (1024, 512, 512, 512)
        self.layers = nn.ModuleList()
        last_dim = state_dim
        for i in range(len(self.hidden)):
            self.layers.append(nn.Linear(last_dim, self.hidden[i]))
            kaiming_normal(self.layers[i])
            last_dim = self.hidden[i]
        self.action_layer = nn.Linear(last_dim, param_dim)
        xavier_init(self.action_layer)
    
    def forward(self, s):
        for i in range(len(self.layers)):
            s = F.tanh(self.layers[i](s))
        s = self.action_layer(s)
        return F.tanh(s)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, param_dim) -> None:
        super(Critic, self).__init__()
        def init_layers():
            layers = nn.ModuleList()
            hidden = (1024, 512, 512, 512)
            last_dim = state_dim + action_dim + param_dim
            for i in range(len(hidden)):
                layers.append(nn.Linear(last_dim, hidden[i]))
                kaiming_normal(layers[i])
                last_dim = hidden[i]
            layers.append(nn.Linear(last_dim, 1))
            kaiming_normal(layers[-1])
            return layers
        self.layers_q1 = init_layers()
        self.layers_q2 = init_layers()

    def forward(self, s, a, p):
        input = torch.cat((s,a,p), dim=1)
        q1 = input
        q2 = input

        for i in range(len(self.layers_q1)):
            q1 = F.relu(self.layers_q1[i](q1))
        
        for i in range(len(self.layers_q2)):
            q2 = F.relu(self.layers_q2[i](q2))
        return q1, q2
    
    def Q1(self, s, a, p):
        input = torch.cat((s,a,p), dim=1)
        negative_slope = 0.01
        q1 = input

        for i in range(len(self.layers_q1)):
            q1 = F.leaky_relu(self.layers_q1[i](q1), negative_slope)
        return q1
        

class TD3:
    def __init__(self, 
        state_dim, 
        action_dim, 
        param_dim,
        tau = 0.01,
        adam_betas=(0.95, 0.999),
        epsilon_initial=1.0,
        epsilon_final=0.01,
        epsilon_steps=10000,
        batch_size=256,
        gamma=0.99,
        beta=0.5,
        tau_actor=0.005,  # Polyak averaging factor for updating target weights
        tau_critic=0.005,
        replay_memory_size=100000,
        initial_memory_threshold=5120,
        loss_func=F.mse_loss, 
        ):
        self.device = torch.device('cuda:1')
        self.tau = tau
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta = beta
        self.tau_actor = tau_actor
        self.tau_critic = tau_critic
        self.action_max = torch.from_numpy(np.ones((action_dim,))).float().to(self.device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        self.action_parameter_max_numpy = np.array([1]*param_dim)
        self.action_parameter_min_numpy = np.array([-1]*param_dim)
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)
        self.replay_memory = Memory(replay_memory_size, (state_dim,), (action_dim+param_dim,), next_actions=False)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.initial_memory_threshold = initial_memory_threshold
        self.loss_func = loss_func
        self.epsilon = epsilon_initial
        self._seed(1)

        self._step = 0
        self._episode = 0
        self.updates = 0
        
        self.policy_noise = 0.01
        self.noise_clip = 0.5

        self.actor = Actor(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.actor_param = ActorParam(state_dim=state_dim, param_dim=param_dim).to(self.device)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, param_dim=param_dim).to(self.device)
        self.actor_target = Actor(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, param_dim=param_dim).to(self.device)
        self.actor_param_target = ActorParam(state_dim=state_dim, param_dim=param_dim).to(self.device)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr = 1e-4, betas=adam_betas)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-4, betas=adam_betas)
        self.opt_actor_param = torch.optim.Adam(self.actor_param.parameters(), lr=1e-4, betas=adam_betas)
        hard_update_target_network(self.actor, self.actor_target)
        hard_update_target_network(self.critic, self.critic_target)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_target.eval()
        self.critic_target.eval()

    def choose_action(self, state, max=False):
        with torch.no_grad():
            state = torch.unsqueeze(torch.from_numpy(state), dim=0).to(self.device)
            all_actions, all_action_parameters = self.actor.forward(state), self.actor_param(state)
            all_actions = all_actions.detach().cpu().data.numpy()
            all_action_parameters = all_action_parameters.detach().cpu().data.numpy()
            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            if self.np_random.uniform() < self.epsilon and max == False:
                all_actions = self.np_random.uniform(size=all_actions.shape)
            action = np.argmax(all_actions)
            action_parameters = all_action_parameters[0][action*2: action*2+2]
        return action, action_parameters, all_actions, all_action_parameters

    def end_episode(self):
        self._episode += 1

        # anneal exploration
        if self._episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    self._episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final
        pass

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    def step(self, state, action, reward, next_state, terminal, optimise=True):
        all_actions, all_action_parameters = action
        self._step += 1
        self._add_sample(state, np.concatenate((all_actions[0].data, all_action_parameters[0].data)).ravel(), reward, next_state, terminal)
        if optimise and self._step >= self.batch_size and self._step >= self.initial_memory_threshold and self._step % 50 == 0:
            self._optimize_td_loss()
    
    def _add_sample(self, state, action, reward, next_state, terminal):
        self.replay_memory.append(state, action, reward, next_state, terminal)

    def _optimize_td_loss(self):        
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and action-parameters
        actions = actions_combined[:,:self.action_dim]
        action_parameters = actions_combined[:, self.action_dim:]
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device)

        # ---------------------- optimize critic ----------------------
        with torch.no_grad():
            noise = (torch.randn_like(action_parameters) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_param = (
                                self.actor_param_target(next_states) + noise  # Noisy on target action
                        ).clamp(-1, 1)
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_action = (
                                self.actor_target(next_states) + noise  # Noisy on target action
                        )

        target_q1, target_q2 = self.critic_target(next_states, smoothed_target_action, smoothed_target_param)
        target_Q = torch.min(target_q1, target_q2)
        target_Q = rewards + (1 - terminals) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(states, actions, action_parameters)
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.opt_critic.zero_grad()
        q_loss.backward()
        self.opt_critic.step()

        # ---------------------- optimise actor ----------------------
        if self._step % 2 == 0:
            actions, action_params = self.actor(states), self.actor_param(states)
            Q_val = -self.critic.Q1(states, actions, action_params).mean()
            self.opt_actor.zero_grad()
            self.opt_actor_param.zero_grad()
            Q_val.backward()
            self.opt_actor.step()
            self.opt_actor_param.step()
            
            soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
            soft_update_target_network(self.critic, self.critic_target, self.tau_critic)
            soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor)

    def start_episode(self):
        pass
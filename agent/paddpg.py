import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from lib.memory import Memory
from torch.autograd import Variable

def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)

def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def normal_init(layer, init_std):
    nn.init.normal_(layer.weight.data, std=init_std)
    nn.init.zeros_(layer.bias.data)

def kaiming_normal(layer):
    nn.init.kaiming_normal_(layer.weight.data, nonlinearity='leaky_relu')
    nn.init.zeros_(layer.bias.data)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, param_dim) -> None:
        super(Actor, self).__init__()
        init_std = 0.001
        self.hidden = (1024, 512, 512, 512)
        self.layers = nn.ModuleList()
        last_dim = state_dim
        for i in range(len(self.hidden)):
            self.layers.append(nn.Linear(last_dim, self.hidden[i]))
            kaiming_normal(self.layers[i])
            last_dim = self.hidden[i]
        self.action_layer = nn.Linear(last_dim, action_dim)
        self.param_layer = nn.Linear(last_dim, param_dim)
        self.param_pass_layer = nn.Linear(state_dim, param_dim)

        nn.init.normal_(self.action_layer.weight, std=init_std)
        nn.init.zeros_(self.action_layer.bias)
        nn.init.normal_(self.param_layer.weight, std=init_std)
        nn.init.zeros_(self.param_layer.bias)

        nn.init.zeros_(self.param_pass_layer.weight)
        nn.init.zeros_(self.param_pass_layer.bias)
        self.param_layer.weight.requires_grad = False
        self.param_layer.bias.requires_grad = False
    
    def forward(self, s):
        s_ = s
        negative_slope = 0.01
        for i in range(len(self.layers)):
            s_ = F.leaky_relu(self.layers[i](s_), negative_slope)
        action = self.action_layer(s_)
        param = self.param_layer(s_)
        param_pass = self.param_pass_layer(s)
        param += param_pass
        return action, param

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, param_dim) -> None:
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden = (1024, 512, 512, 512)
        last_dim = state_dim + action_dim + param_dim

        for i in range(len(self.hidden)):
            self.layers.append(nn.Linear(last_dim, self.hidden[i]))
            kaiming_normal(self.layers[i])
            last_dim = self.hidden[i]
        self.layers.append(nn.Linear(last_dim, 1))
        kaiming_normal(self.layers[-1])

    def forward(self, s, a,p):
        input = torch.cat((s,a,p), dim=1)
        negative_slope = 0.01
        for i in range(len(self.layers)-1):
            input = F.leaky_relu(self.layers[i](input), negative_slope)
        return self.layers[-1](input)
        

class PADDPG:
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

        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, param_dim=param_dim).to(self.device)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, param_dim=param_dim).to(self.device)
        self.actor_target = Actor(state_dim=state_dim, action_dim=action_dim, param_dim=param_dim).to(self.device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, param_dim=param_dim).to(self.device)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr = 1e-4, betas=adam_betas)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-4, betas=adam_betas)
        hard_update_target_network(self.actor, self.actor_target)
        hard_update_target_network(self.critic, self.critic_target)
        self.actor_target.eval()
        self.critic_target.eval()

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU
        if grad_type == "actions":
            max_p = self.action_max.cpu()
            min_p = self.action_min.cpu()
            rnge = self.action_range.cpu()
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max.cpu()
            min_p = self.action_parameter_min.cpu()
            rnge = self.action_parameter_range.cpu()
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            for n in range(grad.shape[0]):
                # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
                index = grad[n] > 0
                grad[n][index] *= (index.float() * (max_p - vals[n]) / rnge)[index]
                grad[n][~index] *= ((~index).float() * (vals[n] - min_p) / rnge)[~index]

        return grad

    def choose_action(self, state, max=False):
        with torch.no_grad():
            state = torch.unsqueeze(torch.from_numpy(state), dim=0).to(self.device)
            all_actions, all_action_parameters = self.actor.forward(state)
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
            pred_next_actions, pred_next_action_parameters = self.actor_target.forward(next_states)
            off_policy_next_val = self.critic_target.forward(next_states, pred_next_actions, pred_next_action_parameters)
            off_policy_target = rewards + (1 - terminals) * self.gamma * off_policy_next_val
            target = off_policy_target

        y_expected = target
        y_predicted = self.critic.forward(states, actions, action_parameters)
        loss_critic = self.loss_func(y_predicted, y_expected)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        # ---------------------- optimise actor ----------------------
        # 1 - calculate gradients from critic
        with torch.no_grad():
            actions, action_params = self.actor(states)
            action_params = torch.cat((actions, action_params), dim=1)
        action_params.requires_grad = True
        Q_val = self.critic(states, action_params[:, :self.action_dim], action_params[:, self.action_dim:]).mean()
        self.critic.zero_grad()
        Q_val.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # 2 - apply inverting gradients and combine with gradients from actor
        actions, action_params = self.actor(Variable(states))
        action_params = torch.cat((actions, action_params), dim=1)
        delta_a[:, self.action_dim:] = self._invert_gradients(delta_a[:, self.action_dim:].cpu(), action_params[:, self.action_dim:].cpu(), grad_type="action_parameters", inplace=True)
        delta_a[:, :self.action_dim] = self._invert_gradients(delta_a[:, :self.action_dim].cpu(), action_params[:, :self.action_dim].cpu(), grad_type="actions", inplace=True)
        out = -torch.mul(delta_a, action_params)
        self.actor.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        self.opt_actor.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.critic, self.critic_target, self.tau_critic)

    def start_episode(self):
        pass
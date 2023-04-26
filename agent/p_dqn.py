import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from lib.memory import Memory
from torch.autograd import Variable

def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


class QActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_parameter_dim, hidden_layers=(1024,512,256)):
        super(QActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_parameter_dim = action_parameter_dim
        
        self.layers = nn.ModuleList()
        input_size = self.state_dim + self.action_parameter_dim
        last_hidden_layer_dim = input_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            for i in range(nh):
                self.layers.append(nn.Linear(last_hidden_layer_dim, hidden_layers[i]))
                last_hidden_layer_dim = hidden_layers[i]
        self.layers.append(nn.Linear(last_hidden_layer_dim, self.action_dim))
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='relu')
            nn.init.zeros_(self.layers[i].bias)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        # implement forward

        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            x = F.relu(self.layers[i](x))
        Q = self.layers[-1](x)
        return Q

class QActorDual(nn.Module):
    def __init__(self, state_dim, action_dim, action_parameter_dim, hidden_layers=(1024,512,256)):
        super(QActorDual, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_parameter_dim = action_parameter_dim
        
        self.layers = nn.ModuleList()
        input_size = self.state_dim + self.action_parameter_dim
        last_hidden_layer_dim = input_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            for i in range(nh):
                self.layers.append(nn.Linear(last_hidden_layer_dim, hidden_layers[i]))
                last_hidden_layer_dim = hidden_layers[i]
        for i in range(0, len(self.layers)):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='relu')
            nn.init.zeros_(self.layers[i].bias)
        self.fc_v = nn.Linear(last_hidden_layer_dim, 1)
        self.fc_a = nn.Linear(last_hidden_layer_dim, action_dim)

    def forward(self, state, action_parameters):
        # implement forward
        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers-1):
            x = F.relu(self.layers[i](x))
        A = self.fc_a(F.relu(self.layers[num_layers-1](x)))
        V = self.fc_v(F.relu(self.layers[num_layers-1](x)))
        Q = V + A - A.mean(1, keepdim=True)
        return Q

class ParamActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_parameter_dim, hidden_layers=(1024,512,256, 256, 256)):
        super(ParamActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_parameter_dim = action_parameter_dim

        self.layers = nn.ModuleList()
        last_hidden_layer_dim = self.state_dim
        if hidden_layers is not None:
            nh = len(hidden_layers)
            for i in range(nh):
                self.layers.append(nn.Linear(last_hidden_layer_dim, hidden_layers[i]))
                last_hidden_layer_dim = hidden_layers[i]
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='relu')
            nn.init.zeros_(self.layers[i].bias)

        self.action_parameters_output_layer = nn.Linear(last_hidden_layer_dim, self.action_parameter_dim)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_dim, self.action_parameter_dim)
        for i in range(0, len(self.layers)):
            nn.init.normal_(self.layers[i].weight, 0.1)
            nn.init.zeros_(self.layers[i].bias)
        nn.init.normal_(self.action_parameters_output_layer.weight, 0.1)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers-1):
            x = F.relu(self.layers[i](x))
        x = F.tanh(self.layers[num_hidden_layers-1](x))
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)
        
        return F.sigmoid(action_params)


class PDQNAgent():
    def __init__(self, observation_space_dim, action_space_dim, double_space = False, actor_type = 'dqn'):
        super(PDQNAgent, self).__init__()
        self.device = torch.device('cuda:1')
        self.num_actions = range(action_space_dim)
        self.action_parameter_dim = action_space_dim
        self.action_space_dim = action_space_dim
        if double_space:
            self.action_parameter_dim *= 2

        self.epsilon = 0.9
        self._step = 0
        self._update = 0
        self.initial_memory_threshold = 256
        self.batch_size = 256
        self.tau_actor = 0.01
        self.tau_actor_param = 0.001
        self.gamma = 0.95
        self._episode = 0
        self.epsilon = 1
        self.epsilon_initial = 1
        self.epsilon_final = 0.05
        self.epsilon_steps = 10000        
        # init memory
        replay_memory_size = 100000
        self.replay_memory = Memory(replay_memory_size, (observation_space_dim,), (1+self.action_parameter_dim,), next_actions=False)
        # init actor 
        if actor_type == 'dqn':
            self.actor = QActor(state_dim=observation_space_dim, action_dim=action_space_dim, action_parameter_dim=self.action_parameter_dim).to(self.device)
            self.actor_target = QActor(state_dim=observation_space_dim, action_dim=action_space_dim, action_parameter_dim=self.action_parameter_dim).to(self.device)
        else:
            self.actor = QActorDual(state_dim=observation_space_dim, action_dim=action_space_dim, action_parameter_dim=self.action_parameter_dim).to(self.device)
            self.actor_target = QActorDual(state_dim=observation_space_dim, action_dim=action_space_dim, action_parameter_dim=self.action_parameter_dim).to(self.device)
        hard_update_target_network(self.actor, self.actor_target)
        # init actor param
        self.actor_param = ParamActor(state_dim=observation_space_dim, action_dim=action_space_dim, action_parameter_dim=self.action_parameter_dim).to(self.device)
        self.actor_param_target = ParamActor(state_dim=observation_space_dim, action_dim=action_space_dim, action_parameter_dim=self.action_parameter_dim).to(self.device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.loss_func = F.mse_loss
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=0.00001)

    def act(self, state, max=False):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)
            # eplison greedy
            rnd = np.random.uniform()
            if rnd < self.epsilon and max == False:
                action = np.random.choice(self.num_actions)
            else:
                # select maximum action
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)

            all_action_parameters = all_action_parameters.cpu().data.numpy()
            if self.action_parameter_dim > self.action_space_dim:
                action_parameters = (all_action_parameters[action*2], all_action_parameters[action*2+1])
            else:
                action_parameters = [all_action_parameters[action]]
        return action, action_parameters, all_action_parameters
    
    def step(self, state, action, reward, next_state, terminal):
        act, all_action_parameters = action
        self._step += 1

        # self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        self._add_sample(state, np.concatenate(([act],all_action_parameters)).ravel(), reward, next_state, terminal=terminal)
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
    
    def _add_sample(self, state, action, reward, next_state, terminal):
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # Compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_Q.backward()

        self.actor_optimiser.step()

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        Q = self.actor(states, action_params)
        Q_val = Q
        Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.actor.zero_grad()
        Q_loss.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        action_params = self.actor_param(Variable(states))
        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final
    
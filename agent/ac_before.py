import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.distributions import Categorical
from collections import namedtuple

# Hyper Parameters for Actor
GAMMA = 0.9 # discount factor

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = F.softmax(self.fc4(x))
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Actor(object):
    def __init__(self, env):
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_dim1 = env.action_dim1
        self.action_dim2 = env.action_dim2

        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)

        # self.network_ = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim2)
        # self.optimizer_ = torch.optim.Adam(self.network_.parameters(), lr=0.15)


    def choose_action(self, observation, max=False):
        observation =  torch.from_numpy(observation).float().unsqueeze(0)
        network_output = self.network(observation)
        print(observation)
        # print(network_output)
        # network_output_ = self.network_.forward(observation)
        with torch.no_grad():
            prob_weights = network_output
            # prob_weights_ = F.softmax(network_output_, dim=0).cuda().data.cpu()
        if max:
            action = torch.argmax(prob_weights)
            # action_ = torch.argmax(prob_weights_)
        else:
            action =  Categorical(prob_weights).sample()  
            # action_ = np.random.choice(range(prob_weights_.shape[0]),
            #                       p=prob_weights_.numpy())
        return action.item()

    def learn(self, state, action, td_error):
        observation =  torch.from_numpy(state).float().unsqueeze(0)
        action = torch.LongTensor([action])
        softmax_input = self.network(observation)
        l=torch.nn.NLLLoss()
        neg_log_prob=l(torch.log(softmax_input), action)
        loss_a = neg_log_prob * td_error
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()


class QNetwork(nn.Module):
    def __init__(self, state_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)   

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Critic(object):
    def __init__(self, env):
        self.state_dim = env.state_dim

        self.network = QNetwork(state_dim=self.state_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.04)
        self.loss_func = nn.MSELoss()

    def train_Q_network(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state), torch.FloatTensor(next_state)
        v = self.network.forward(s)     
        v_ = self.network.forward(s_).detach()
        loss_q = self.loss_func(reward + GAMMA * v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()
        print(v_, v)
        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v

        return td_error






class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dim, 128)

        self.action1 = nn.Linear(128, 256)
        self.action2 = nn.Linear(256, action_dim)

        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)
        x = F.relu(self.action1(x))
        action_prob = F.softmax(self.action2(x), dim=-1)
        return action_prob, state_values
    
    def choose_action(self, observation, store=False, max=False):
        observation = torch.FloatTensor(observation)
        probs, state_value = self(observation)
        if max == False:
            m = Categorical(probs)
            action = m.sample()
        else:
            action = torch.argmax(probs)
        if store:
            self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def learn(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()

                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    




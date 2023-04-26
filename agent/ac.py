import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

GAMMA = 0.8

class PGNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_size)
        nn.init.normal_(self.fc1.weight.data, 0, 1)
        nn.init.constant_(self.fc1.bias.data, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 1)
        nn.init.constant_(self.fc2.bias.data, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 1)
        nn.init.constant_(self.fc3.bias.data, 0.01)
        self.loss_func =  nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return  torch.softmax(self.fc3(x), dim=1)

    def learn(self, state, action, td_error):
        observation = torch.from_numpy(state).float().unsqueeze(0)
        softmax_input = self(observation)
        action = torch.LongTensor([action])
        neg_log_prob = self.loss_func(softmax_input, action)
        loss_a = neg_log_prob * td_error
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()
    
    def choose_action(self, s, max=False):
        observation =  torch.from_numpy(s).float().unsqueeze(0)
        probs = self(observation).detach()
        
        # if max:
        #     action = torch.argmax(probs)
        # else:
        m = Categorical(probs)
        action = m.sample()
        # print(action)
        # print(action)
        return action.item()

class Actor():
    def __init__(self, env):
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_dim1 = env.action_dim1
        self.action_dim2 = env.action_dim2

        self.network1 = PGNetwork(self.state_dim, self.action_dim1)
        self.network2 = PGNetwork(self.state_dim, self.action_dim2)
    
    def choose_action(self, s, max=False):
        action1 = self.network1.choose_action(s)
        action2 = self.network2.choose_action(s)
        return action1, action2

    def learn(self, state, action, td_error):
        # print(td_error)
        self.network1.learn(state, action[0], td_error)
        self.network2.learn(state, action[1], td_error)

class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.loss_func = nn.MSELoss()
        nn.init.normal_(self.fc1.weight.data, 0, 1)
        nn.init.constant_(self.fc1.bias.data, 0.1)
        nn.init.normal_(self.fc2.weight.data, 0, 1)
        nn.init.constant_(self.fc2.bias.data, 0.1)
        nn.init.normal_(self.fc3.weight.data, 0, 1)
        nn.init.constant_(self.fc3.bias.data, 0.1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def learn(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state), torch.FloatTensor(next_state)
        v = self(s)
        v_ = self(s_).detach()
        loss_q = self.loss_func(reward + GAMMA *v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()
        # print(v, v_)
        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v
        return td_error.item()

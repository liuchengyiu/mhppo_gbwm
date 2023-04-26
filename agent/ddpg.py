import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)

# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mu_bound):
        super(Actor, self).__init__()
        self.mu_bound = torch.FloatTensor(mu_bound)

        # layer
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        nn.init.normal_(self.layer_1.weight, 0., 0.3)
        nn.init.normal_(self.layer_2.weight, 0., 0.3)
        nn.init.constant_(self.layer_1.bias, 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)
        # self.layer_1.weight.data.normal_(0.,0.3)
        # self.layer_1.bias.data.fill_(0.1)
        self.output = nn.Linear(32, action_dim)
        self.output.weight.data.normal_(0.,0.3)
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        a = torch.relu(self.layer_1(s))
        a = torch.tanh(self.layer_2(a))
        a = F.sigmoid(self.output(a))
        return a


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 30
        # layer
        self.layer_1 = nn.Linear(state_dim, n_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)
        
        self.layer_2 = nn.Linear(action_dim, n_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)
        
        self.output = nn.Linear(n_layer, 1)

    def forward(self, s, a):
        
        s = self.layer_1(s)
        a = self.layer_2(a)
        q_val = self.output(torch.relu(s+a))
        return q_val


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, mu_bound, replacement,memory_capacity=1000,gamma=0.9,lr_a=0.001, lr_c=0.002,batch_size=32) :
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        # 记忆库
        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim, mu_bound)
        self.actor_target = Actor(state_dim, action_dim, mu_bound)
        # 定义 Critic 网络
        self.critic = Critic(state_dim,action_dim)
        self.critic_target = Critic(state_dim,action_dim)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    def sample(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices, :] 

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        s = s.reshape((-1, self.state_dim))
        action = self.actor(s)
        return action.detach()[0].numpy()

    def learn(self):

        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                a = self.actor.state_dict()[al[0]+'.weight']
                al[1].weight.data.mul_((1-tau))
                al[1].weight.data.add_(tau * self.actor.state_dict()[al[0]+'.weight'])
                al[1].bias.data.mul_((1-tau))
                al[1].bias.data.add_(tau * self.actor.state_dict()[al[0]+'.bias'])
            for cl in c_layers:
                cl[1].weight.data.mul_((1-tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0]+'.weight'])
                cl[1].bias.data.mul_((1-tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0]+'.bias'])
            
        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0]+'.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0]+'.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0]+'.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0]+'.bias']
            
            self.t_replace_counter += 1

        # 从记忆库中采样bacth data
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim])
        bs_ = torch.FloatTensor(bm[:,-self.state_dim:])
        
        # 训练Actor
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()
        
        # 训练critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target,q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor_eval.pt"))
        torch.save(self.actor_target.state_dict(), os.path.join(path, "actor_target.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic_eval.pt"))
        torch.save(self.critic_target.state_dict(), os.path.join(path, "critic_target.pt"))
    
    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor_eval.pt")))
        self.actor_target.load_state_dict(torch.load(os.path.join(path, "actor_target.pt")))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic_eval.pt")))
        self.critic_target.load_state_dict(torch.load(os.path.join(path, "critic_target.pt")))
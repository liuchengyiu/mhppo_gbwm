import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Beta, Gamma, StudentT, FisherSnedecor
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def kaiming_normal(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    nn.init.zeros_(layer.bias)

def xavier_init(layer):
    tanh_gain = nn.init.calculate_gain('tanh')
    nn.init.xavier_normal_(layer.weight, tanh_gain)
    nn.init.zeros_(layer.bias)

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorDis(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorDis, self).__init__()
        self.lstm = nn.LSTM(state_dim, 256, 6)
        hidden_layer = (256, 256)
        self.layers = nn.ModuleList()
        last_layer_dim = 256
        for i in range(len(hidden_layer)):
            self.layers.append(nn.Linear(last_layer_dim, hidden_layer[i]))
            xavier_init(self.layers[i])
            last_layer_dim = hidden_layer[i]
        self.layers.append(nn.Linear(last_layer_dim, action_dim))
        xavier_init(self.layers[-1])
    
    def forward(self, x):
        x, _ = self.lstm(x)
        for i in range(len(self.layers)-1):
            x = F.tanh(self.layers[i](x))
        x = F.softmax(self.layers[-1](x), dim=-1)
        return x

class ActorCon(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCon, self).__init__()
        self.lstm = nn.LSTM(state_dim, 256, 6)
        hidden_layer = (256, 256)
        self.layers = nn.ModuleList()
        last_layer_dim = 256
        self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True) 
        for i in range(len(hidden_layer)):
            self.layers.append(nn.Linear(last_layer_dim, hidden_layer[i]))
            xavier_init(self.layers[i])
            last_layer_dim = hidden_layer[i]
        self.layers.append(nn.Linear(last_layer_dim, action_dim))
        xavier_init(self.layers[-1])
    
    def forward(self, x):
        x, _ = self.lstm(x)
        for i in range(len(self.layers)):
            x = F.tanh(self.layers[i](x))
        return x
    
    def get_dist(self, x):
        mean = self.forward(x)
        log_std = self.log_std.expand_as(mean) 
        std = torch.exp(log_std) 
        dist = Normal(mean, std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(state_dim, 256, 6)
        hidden_layer = (256, 256)
        self.layers = nn.ModuleList()
        last_layer_dim = 256
        for i in range(len(hidden_layer)):
            self.layers.append(nn.Linear(last_layer_dim, hidden_layer[i]))
            xavier_init(self.layers[i])
            last_layer_dim = hidden_layer[i]
        self.layers.append(nn.Linear(last_layer_dim, 1))
        xavier_init(self.layers[-1])

    def forward(self, s):
        s, _ = self.lstm(s)
        for i in range(len(self.layers)-1):
            s = F.tanh(self.layers[i](s))
        return self.layers[-1](s)

class HPPO:
    def __init__(self, 
                state_dim, 
                action_dim, 
                param_dim, 
                gamma=0.9, 
                epsilon=0.2, 
                torch_device = 0,
                entropy_coef=0.001,
                ac_type = 'normal',
                lr_ad = 3e-4,
                lr_ac = 3e-4,
                lr_c = 3e-4,
                k_epoch = 10,
                batch_size=4096,
                max_train_steps=2e6,
                mini_batch_size=256):
        self.device = torch.device('cuda:{}'.format(torch_device))
        print("mine")
        self.ac_type = ac_type
        self.actor_dis = ActorDis(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.actor_con = ActorCon(state_dim=state_dim, action_dim=param_dim).to(self.device)
        self.critic = Critic(state_dim=state_dim).to(self.device)
        print("Actor Dis param: ", sum(p.numel() for p in self.actor_dis.parameters()))
        print("Actor con param: ", sum(p.numel() for p in self.actor_con.parameters()))
        print("Actor critic param: ", sum(p.numel() for p in self.critic.parameters()))
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.k_epoch = k_epoch
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.lr_ac = lr_ac
        self.lr_ad = lr_ad
        self.lr_c = lr_c
        self.max_train_steps = max_train_steps
        self.lamda = 0.95
        self.opt_actor_con = torch.optim.Adam(self.actor_con.parameters(), lr=lr_ac, eps=1e-5)
        self.opt_actor_dis = torch.optim.Adam(self.actor_dis.parameters(), lr=lr_ad, eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_c,  eps=1e-5)
    
    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), dim=0).to(self.device)

        with torch.no_grad():
            a_prob = self.actor_dis(s)
            p_dist = self.actor_con.get_dist(s)

            a_dist = Categorical(probs=a_prob)
            a = a_dist.sample()
            a_logprob = a_dist.log_prob(a)
            p = p_dist.sample()
            p = torch.clamp(p, -1, 1)
            p_logprob = p_dist.log_prob(p)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0], p.cpu().numpy().flatten(), p_logprob.cpu().numpy().flatten()
    
    def evalate(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), dim=0).to(self.device)
        with torch.no_grad():
            a_prob = self.actor_dis(s).detach().cpu().numpy().flatten()
            p = self.actor_con(s).detach().cpu().numpy().flatten()
            a = np.argmax(a_prob)
        return a, p

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, p, p_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()    
        with torch.no_grad():
            gae = 0
            adv = []
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                # update actor param
                p_dist_now = self.actor_con.get_dist(s[index])
                p_dist_entropy = p_dist_now.entropy().sum(1, keepdim=True)
                p_logprob_now = p_dist_now.log_prob(p[index])
                ratio = torch.exp(p_logprob_now.sum(1, keepdim=True) - p_logprob[index].sum(1, keepdim=True))

                sur1 = adv[index] * ratio
                sur2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_con_loss = -torch.min(sur1, sur2) - self.entropy_coef * p_dist_entropy
                self.opt_actor_con.zero_grad()
                actor_con_loss.mean().backward()
                self.opt_actor_con.step()

                #update actor
                dist_now = Categorical(probs=self.actor_dis(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)
                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                self.opt_actor_dis.zero_grad()
                actor_loss.mean().backward()
                self.opt_actor_dis.step()
                
                # Update critic
                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                self.opt_critic.zero_grad()
                critic_loss.backward()
                self.opt_critic.step()


        # if self.use_lr_decay:  # Trick 6:learning rate Decay
        self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_ac_now = self.lr_ac * (1 - total_steps / self.max_train_steps)
        lr_ad_now = self.lr_ad * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        print(lr_ac_now, lr_ad_now, lr_c_now)
        for p in self.opt_actor_con.param_groups:
            p['lr'] = lr_ac_now
        for p in self.opt_actor_dis.param_groups:
            p['lr'] = lr_ad_now
        for p in self.opt_critic.param_groups:
            p['lr'] = lr_c_now
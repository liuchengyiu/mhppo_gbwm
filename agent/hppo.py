import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from torch.distributions import Beta, Normal


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def kaiming_init(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    nn.init.zeros_(layer.bias)

class ActorDiscrete(nn.Module):
    def __init__(self, args):
        super(ActorDiscrete, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.action_dim)
        # self.fc3 = nn.Linear(args.hidden_width, args.hidden_width)
        # self.fc4 = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        # self.activate_func = [nn.ReLU(), nn.Tanh()][0]  # Trick10: use tanh
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2, gain=0.01)
            # orthogonal_init(self.fc3)
            # orthogonal_init(self.fc4, gain=0.01)

        # if args.use_orthogonal_init:
        #     print("------use_orthogonal_init------")
        #     kaiming_init(self.fc1)
        #     kaiming_init(self.fc2)
        #     kaiming_init(self.fc3)
        #     kaiming_init(self.fc4, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        # s = self.activate_func(self.fc2(s))
        # s = self.activate_func(self.fc3(s))
        a_prob = torch.softmax(self.fc2(s), dim=1)
        return a_prob

class ActorContinue(nn.Module):
    def __init__(self, args):
        super(ActorContinue, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc4 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim*2)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim*2))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))
        s = self.activate_func(self.fc4(s))
        mean = self.max_action * (torch.tanh(self.mean_layer(s)) + 1) / 2 # [-1,1]->[0, 1]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class ActorDC(nn.Module):
    def __init__(self, args):
        super(ActorDC, self).__init__()
        self.comm_layer = nn.ModuleList()
        self.comm_hidd = (512, 512, 256)
        self.continue_layer = nn.ModuleList()
        self.continue_hidden = (256, args.action_dim*2)
        self.discrete_layer = nn.ModuleList()
        self.discrete_hidden = (256, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim*2))
        self.max_action = args.max_action
        # init common layer
        last_hidden_layer_dim = args.state_dim
        for i in range(len(self.comm_hidd)):
            self.comm_layer.append(nn.Linear(last_hidden_layer_dim, self.comm_hidd[i]))
            last_hidden_layer_dim = self.comm_hidd[i]
            kaiming_init(self.comm_layer[i])
        
        # init continue
        for i in range(len(self.continue_hidden)):
            self.continue_layer.append(nn.Linear(last_hidden_layer_dim, self.continue_hidden[i]))
            last_hidden_layer_dim = self.continue_hidden[i]
            orthogonal_init(self.continue_layer[i], gain=0.01)
        
        # init discrete
        for i in range(len(self.discrete_hidden)):
            last_hidden_layer_dim = self.comm_hidd[-1]
            self.discrete_layer.append(nn.Linear(last_hidden_layer_dim, self.discrete_hidden[i]))
            last_hidden_layer_dim = self.continue_hidden[i]
            kaiming_init(self.discrete_layer[i])

    def forward(self, s, singleC=False, singleD=False):
        # encode s
        for i in range(len(self.comm_layer)):
            s = F.relu(self.comm_layer[i](s))
        if not singleC:
            a = s
            for i in range(len(self.discrete_layer)-1):
                a = F.relu(self.discrete_layer[i](a))
            a_prob = torch.softmax(self.discrete_layer[-1](a), dim=1)
            
        if not singleD:
            p = s
            for i in range(len(self.continue_layer)-1):
                p = F.tanh(self.continue_layer[i](p))
            mean = self.max_action * (torch.tanh(self.continue_layer[-1](p)) + 1) / 2

        if singleD:
            return a_prob
        if singleC:
            return mean
        return a_prob, mean

    def get_dist(self, s):
        mean = self.forward(s, singleC=True)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

class HPPO():
    def __init__(self, args):
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = ActorDiscrete(args)
        self.actor_param = ActorContinue(args)
        self.actor_dc = ActorDC(args)
        self.critic = Critic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_actor_param = torch.optim.Adam(self.actor_param.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_actor_dc = torch.optim.Adam(self.actor_dc.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_actor_param = torch.optim.Adam(self.actor_param.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
            self.optimizer_actor_dc = torch.optim.Adam(self.actor_dc.parameters(), lr=self.lr_c)


    def evaluate(self, s, dc=False):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if dc:
            a_prob, p = self.actor_dc(s)
            a_prob = a_prob.detach().numpy().flatten()
            p = p.detach().numpy().flatten()
        else:
            a_prob = self.actor(s).detach().numpy().flatten()
            p = self.actor_param(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a, p

    def choose_action(self, s, dc=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if dc:
            with torch.no_grad():
                a_prob, p = self.actor_dc(s)
                dist = Categorical(probs=a_prob)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
                dist = self.actor_dc.get_dist(s)
                p = dist.sample()
                p = torch.clamp(p, 0, self.max_action)
                p_logprob = dist.log_prob(p)
        else:
            with torch.no_grad():
                dist = Categorical(probs=self.actor(s))
                a = dist.sample()
                a_logprob = dist.log_prob(a)
            with torch.no_grad():
                dist = self.actor_param.get_dist(s)
                p = dist.sample()  # Sample the action according to the probability distribution
                p = torch.clamp(p, 0, self.max_action)  # [-max,max]
                p_logprob = dist.log_prob(p)  # The log probability density of the action
        return a.numpy()[0], a_logprob.numpy()[0], p.numpy().flatten(), p_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps, dc=False):
        s, a, a_logprob, p,  p_logprob, r, r_, s_, s__, dw, done, done_ = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            vs__ = self.critic(s__)
            for v_s, v_s_, v_s__, _r, _r_, _done, _done_ in zip(vs, vs_, vs__, r, r_, done, done_):
                if _done:
                    adv.append(_r-v_s)
                    continue
                if _done_:
                    adv.append(_r-v_s + self.gamma*v_s_)
                    continue
                adv.append(_r-v_s + self.gamma*_r_ + self.gamma**2 * v_s__)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                if dc:
                    dist_now = self.actor_dc.get_dist(s[index])
                    dist_entropy = dist_now.entropy().sum(1, keepdim=True) 
                    p_logprob_now = dist_now.log_prob(p[index])
                    # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                    ratios = torch.exp(p_logprob_now.sum(1, keepdim=True) - p_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                    surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                    # Update actor
                    self.optimizer_actor_dc.zero_grad()
                    actor_loss.mean().backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.actor_dc.parameters(), 0.5)
                    self.optimizer_actor_dc.step()

                    #update Actor
                    dist_now = Categorical(probs=self.actor_dc.forward(s[index], singleD=True))
                    dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                    a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                    # a/b=exp(log(a)-log(b))
                    ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                    surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                    # Update actor
                    self.optimizer_actor_dc.zero_grad()
                    actor_loss.mean().backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.actor_dc.parameters(), 0.5)
                    self.optimizer_actor_dc.step()
                else:
                    #update Actor param
                    dist_now = self.actor_param.get_dist(s[index])
                    dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                    p_logprob_now = dist_now.log_prob(p[index])
                    # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                    ratios = torch.exp(p_logprob_now.sum(1, keepdim=True) - p_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                    surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                    # Update actor
                    self.optimizer_actor_param.zero_grad()
                    actor_loss.mean().backward()
                    # if self.use_grad_clip:  # Trick 7: Gradient clip
                    #     torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), 0.5)
                    self.optimizer_actor_param.step()


                    #update Actor
                    dist_now = Categorical(probs=self.actor(s[index]))
                    dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                    a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                    # a/b=exp(log(a)-log(b))
                    ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                    surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                    # Update actor
                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()
                    # if self.use_grad_clip:  # Trick 7: Gradient clip
                    #     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()

                    # Update critic
                    v_s = self.critic(s[index])
                    critic_loss = F.mse_loss(v_target[index], v_s)
                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

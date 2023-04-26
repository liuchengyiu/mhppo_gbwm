import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done

class ReplayBufferV2:
    def __init__(self, args, device):
        self.device = torch.device('cuda:{}'.format(device))
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.a = np.zeros((args.batch_size, 1))
        self.p = np.zeros((args.batch_size, args.action_dim*2))
        self.p_logprob = np.zeros((args.batch_size, args.action_dim*2))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, p, p_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.p[self.count] = p
        self.p_logprob[self.count] = p_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.a, dtype=torch.long).to(self.device)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        p = torch.tensor(self.p, dtype=torch.float).to(self.device)
        p_logprob = torch.tensor(self.p_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return s, a, a_logprob, p, p_logprob, r, s_, dw, done

class ReplayBufferV2_:
    def __init__(self, args, device):
        self.device = torch.device('cuda:{}'.format(device))
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.a = np.zeros((args.batch_size, 1))
        self.p = np.zeros((args.batch_size, args.action_dim))
        self.p_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, p, p_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.p[self.count] = p
        self.p_logprob[self.count] = p_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.a, dtype=torch.long).to(self.device)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        p = torch.tensor(self.p, dtype=torch.float).to(self.device)
        p_logprob = torch.tensor(self.p_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return s, a, a_logprob, p, p_logprob, r, s_, dw, done

class ReplayBufferV3:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.s__ = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.a = np.zeros((args.batch_size, 1))
        self.p = np.zeros((args.batch_size, args.action_dim*2))
        self.p_logprob = np.zeros((args.batch_size, args.action_dim*2))
        self.r = np.zeros((args.batch_size, 1))
        self.r_ = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.done_ = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, p, p_logprob, r, r_, s_, s__, dw, done, done_):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.p[self.count] = p
        self.p_logprob[self.count] = p_logprob
        self.r[self.count] = r
        self.r_[self.count] = r_
        self.s_[self.count] = s_
        self.s__[self.count] = s__
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.done_[self.count] = done_
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        p = torch.tensor(self.p, dtype=torch.float)
        p_logprob = torch.tensor(self.p_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        r_ = torch.tensor(self.r_, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        s__ = torch.tensor(self.s__, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        done_ = torch.tensor(self.done_, dtype=torch.float)

        return s, a, a_logprob, p, p_logprob, r, r_, s_, s__, dw, done, done_
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init
import numpy as np
import math

import sys
import datetime
def print_now(cmd):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('%s %s' % (time_now, cmd))
    sys.stdout.flush()

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent   = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)
    @property
    def is_recurrent(self):
        return self._recurrent
    
    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size if self._recurrent else 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T*N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x     = x.view(T, N, x.size(1))
            masks = mask.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs*masks[i])
                output.append(hx)
            # 
            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T*N, -1)
        return x, hxs 

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        # Uniform Distribution bounds: 
        #     U(-1/sqrt(p), 1/sqrt(p))
        self.lowerU          = -1.0 / math.sqrt(in_features) # 
        self.upperU          =  1.0 / math.sqrt(in_features) # 
        self.sigma_0         = std_init
        self.sigma_ij_in     = self.sigma_0 / math.sqrt(self.in_features)
        self.sigma_ij_out    = self.sigma_0 / math.sqrt(self.out_features)

        """
        Registre_Buffer: Adds a persistent buffer to the module.
            A buffer that is not to be considered as a model parameter -- like "running_mean" in BatchNorm
            It is a "persistent state" and can be accessed as attributes --> self.weight_epsilon
        """
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        self.weight_mu.data.uniform_(self.lowerU, self.upperU)
        self.weight_sigma.data.fill_(self.sigma_ij_in)

        self.bias_mu.data.uniform_(self.lowerU, self.upperU)
        self.bias_sigma.data.fill_(self.sigma_ij_out)

    def sample_noise(self):
        eps_in  = self.func_f(self.in_features)
        eps_out = self.func_f(self.out_features)
        # Take the outter product 
        """
            >>> v1 = torch.arange(1., 5.) [1, 2, 3, 4]
            >>> v2 = torch.arange(1., 4.) [1, 2, 3]
            >>> torch.ger(v1, v2)
            tensor([[  1.,   2.,   3.],
                    [  2.,   4.,   6.],
                    [  3.,   6.,   9.],
                    [  4.,   8.,  12.]])
        """
        eps_ij = eps_out.ger(eps_in)
        self.weight_epsilon.copy_(eps_ij)
        self.bias_epsilon.copy_(eps_out)

    def func_f(self, n): # size
        # sign(x) * sqrt(|x|) as in paper
        x = torch.rand(n)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma*self.weight_epsilon, 
                               self.bias_mu   + self.bias_sigma  *self.bias_epsilon)

        else:
            return F.linear(x, self.weight_mu,
                               self.bias_mu)
        

    
class DQN(nn.Module):
    def __init__(self, num_inputs, hidden_size=512, num_actions=1, use_duel=False, use_noisy_net=False):
        super(DQN, self).__init__()
        init_ = lambda m: init(m, 
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0), 
                               nn.init.calculate_gain('relu'))
        init2_ = lambda m: init(m, 
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))        
        self.use_duel = use_duel

        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))

        if use_noisy_net:
            Linear = NoisyLinear
        else:
            Linear = nn.Linear

        if self.use_duel:
            self.val_fc        = Linear(32*7*7, hidden_size)
            self.val           = Linear(hidden_size, 1)
            self.adv_fc        = Linear(32*7*7, hidden_size)
            self.adv           = Linear(hidden_size, num_actions)
            if not use_noisy_net:
                self.val_fc    = init_(self.val_fc)
                self.adv_fc    = init_(self.adv_fc)
                self.val       = init2_(self.val)
                self.adv       = init2_(self.adv)

        else:
            self.fc                = Linear(32*7*7, hidden_size)
            self.critic_linear     = Linear(hidden_size, num_actions)
            if not use_noisy_net:
                self.fc            = init_(self.fc)
                self.critic_linear = init2_(self.critic_linear)

        self.train()


    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        if self.use_duel:
            val = self.val(F.relu(self.val_fc(x)))
            adv = self.adv(F.relu(self.adv_fc(x)))
            y   = val + adv - adv.mean()
        else:
            x = F.relu(self.fc(x))
            y = self.critic_linear(x)            
        return y

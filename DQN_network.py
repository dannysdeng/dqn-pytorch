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
        self.use_noisy_net = use_noisy_net

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
    def sample_noise(self):
        if self.use_noisy_net:
            if self.use_duel:
                self.val_fc.sample_noise()
                self.val.sample_noise()
                self.adv_fc.sample_noise()
                self.adv.sample_noise()
            else:
                self.fc.sample_noise()
                self.critic_linear.sample_noise()


class C51(nn.Module):
    def __init__(self, num_inputs, hidden_size=512, num_actions=4, 
                       use_duel=False, use_noisy_net=False, atoms=51, vmin=-10, vmax=10, use_qr_c51=False):
        super(C51, self).__init__()
        self.atoms = atoms
        self.vmin  = vmin
        self.vmax  = vmax
        self.num_actions   = num_actions
        self.use_duel      = use_duel
        self.use_noisy_net = use_noisy_net
        self.use_qr_c51    = use_qr_c51


        init_ = lambda m: init(m, 
                               nn.init.kaiming_uniform_,
                               lambda x: nn.init.constant_(x, 0), 
                               nonlinearity='relu',
                               mode='fan_in')
        init2_ = lambda m: init(m, 
                               nn.init.kaiming_uniform_,
                               lambda x: nn.init.constant_(x, 0), 
                               nonlinearity='relu',
                               mode='fan_in')        


        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))    

        if use_noisy_net:
            Linear = NoisyLinear
        else:
            Linear = nn.Linear

        self.fc1 = Linear(32*7*7,      hidden_size)
        self.fc2 = Linear(hidden_size, num_actions*atoms)
    
        if self.use_duel:
            self.val_fc = Linear(32*7*7, hidden_size)
            self.val    = Linear(hidden_size, atoms)

        # Param init
        if not use_noisy_net:
            self.fc1 = init_(self.fc1)
            self.fc2 = init2_(self.fc2)
            if self.use_duel:
                self.val_fc = init_(self.val_fc)
                self.val    = init2_(self.val)

     

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        if self.use_duel:
            val_x    = F.relu(self.val_fc(x))
            values   = self.val(val_x).unsqueeze(1) # from batch x atoms to batch x 1 x atoms

            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x_batch = x.view(-1, self.num_actions, self.atoms)

            duel = values + x_batch - x_batch.mean(1, keepdim=True)
            if self.use_qr_c51:
                y = duel
            else:
                y = F.log_softmax(duel, dim = 2).exp() # y is of shape [batch x action x atoms]             
        else:
            # A Tensor of shape [batch x actions x atoms].
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x_batch = x.view(-1, self.num_actions, self.atoms)
            if self.use_qr_c51:
                y = x_batch
            else:
                y = F.log_softmax(x_batch, dim=2).exp() # y is of shape [batch x action x atoms]            
            
        return y

    def sample_noise(self):
        if self.use_noisy_net:
            if self.use_duel:
                self.fc1.sample_noise()
                self.fc2.sample_noise()
                self.val_fc.sample_noise()
                self.val.sample_noise()
            else:
                self.fc1.sample_noise()
                self.fc2.sample_noise()    

class IQN_C51(nn.Module):
    def __init__(self, num_inputs, hidden_size=512, num_actions=4, 
                       use_duel=False, use_noisy_net=False):
        super(IQN_C51, self).__init__()
        self.num_actions   = num_actions
        self.use_duel      = use_duel
        self.use_noisy_net = use_noisy_net
        self.quantile_embedding_dim = 64
        self.pi = np.pi


        init_ = lambda m: init(m, 
                               nn.init.kaiming_uniform_,
                               lambda x: nn.init.constant_(x, 0), 
                               gain=nn.init.calculate_gain('relu'),
                               mode='fan_in')
        init2_ = lambda m: init(m, 
                               nn.init.kaiming_uniform_,
                               lambda x: nn.init.constant_(x, 0), 
                               gain=nn.init.calculate_gain('relu'),
                               mode='fan_in')        


        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))    

        if use_noisy_net:
            Linear = NoisyLinear
        else:
            Linear = nn.Linear
        # ----------------------------------------------------------------------------
        # self.fc1 = Linear(32*7*7, hidden_size)
        self.fc2 = Linear(hidden_size, num_actions*1)
        # ----------------------------------------------------------------------------
        Atari_Input = torch.FloatTensor(1, num_inputs, 84, 84)
        temp_fea    = self.conv3(self.conv2(self.conv1(Atari_Input)))
        temp_fea    = temp_fea.view(temp_fea.size(0), -1)
        state_net_size = temp_fea.size(1)
        del Atari_Input
        del temp_fea

        self.quantile_fc0 = nn.Linear(self.quantile_embedding_dim, state_net_size)
        self.quantile_fc1 = nn.Linear(state_net_size, hidden_size)
        # ----------------------------------------------------------------------------
        if self.use_duel:
            self.quantile_fc_value  = Linear(hidden_size, 1)

        # Param init
        if not use_noisy_net:
            self.quantile_fc0 = init2_(self.quantile_fc0)
            self.quantile_fc1 = init2_(self.quantile_fc1)
            self.fc2 = init2_(self.fc2)
            if self.use_duel:
                self.quantile_fc_value = init2_(self.quantile_fc_value)


    def forward(self, x, num_quantiles):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        BATCH_SIZE     = x.size(0)
        state_net_size = x.size(1)

        tau = torch.FloatTensor(BATCH_SIZE * num_quantiles, 1).to(x)
        tau.uniform_(0, 1)
        # ----------------------------------------------------------------------------------------------
        quantile_net = torch.FloatTensor([i for i in range(1, 1+self.quantile_embedding_dim)]).to(x) 
        # -------------------------------------------------------------------------------------------------
        tau_expand   = tau.unsqueeze(-1).expand(-1, -1, self.quantile_embedding_dim) # [Batch*Np x 1 x 64]
        quantile_net = quantile_net.view(1, 1, -1) # [1 x 1 x 64] -->         [Batch*Np x 1 x 64]
        quantile_net = quantile_net.expand(BATCH_SIZE*num_quantiles, 1, self.quantile_embedding_dim)
        cos_tau      = torch.cos(quantile_net * self.pi * tau_expand)       # [Batch*Np x 1 x 64]
        cos_tau      = cos_tau.squeeze(1)                                   # [Batch*Np x 64]
        # -------------------------------------------------------------------------------------------------
        out          = F.relu(self.quantile_fc0(cos_tau))                   # [Batch*Np x feaSize]
        # fea_tile     = torch.cat([x]*num_quantiles, dim=0)
        fea_tile     = x.unsqueeze(1).expand(-1, num_quantiles, -1)         # [Batch x Np x feaSize]
        out          = out.view(BATCH_SIZE, num_quantiles, -1)              # [Batch x Np x feaSize]
        product      = (fea_tile * out).view(BATCH_SIZE*num_quantiles, -1)
        combined_fea = F.relu(self.quantile_fc1(product))                   # (Batch*atoms, 512)

        if self.use_duel:
            values   = self.quantile_fc_value(combined_fea) # from [batch*atoms x 1] to [Batch x 1 x Atoms]
            values   = values.view(-1, num_quantiles).unsqueeze(1)

            x        = self.fc2(combined_fea)
            x_batch  = x.view(BATCH_SIZE, num_quantiles, self.num_actions)
            # After transpose, x_batch becomes [batch x actions x atoms]
            x_batch  = x_batch.transpose(1, 2).contiguous()
            action_component = x_batch - x_batch.mean(1, keepdim=True)

            duel_y = values + action_component
            y      = duel_y
        else:
            x = self.fc2(combined_fea)
            #           [batch x atoms x actions].
            y = x.view(BATCH_SIZE, num_quantiles, self.num_actions)    
            # output should be # A Tensor of shape [batch x actions x atoms].
            y = y.transpose(1, 2).contiguous()
        # ------------------------------------------------------------------------------------------------ #
        return y, tau # [batch x actions x atoms]

    def sample_noise(self):
        if self.use_noisy_net:
            if self.use_duel:
                self.fc2.sample_noise()
                self.quantile_fc_value.sample_noise()
            else:
                self.fc2.sample_noise()
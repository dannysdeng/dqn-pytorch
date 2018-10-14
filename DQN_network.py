import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init

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
    
class DQN(nn.Module):
    def __init__(self, num_inputs, hidden_size=512, num_actions=1):
        super(DQN, self).__init__()
        init_ = lambda m: init(m, 
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0), 
                               nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        self.fc    = init_(nn.Linear(32*7*7, hidden_size))


        init_ = lambda m: init(m, 
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, num_actions))
        self.train()

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.critic_linear(x)

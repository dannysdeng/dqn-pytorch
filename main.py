"""
Great tutorial: https://github.com/qfettes/DeepRL-Tutorials
"""
import copy
import glob
import os
import time
from collections import deque
import random
import argparse

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc


""" A2C specific arguments """
#import algo
#from arguments import get_args

# from envs  import make_vec_envs
# from model import Policy
# from storage import RolloutStorage
# from utils import get_vec_normalize

# DQN specific arguments
from DQN_network import DQN, C51
from replay_memory import ReplayMemory, PrioritizedReplayBuffer
from utils import init
from env import make_vec_envs
from baselines.common.schedules import LinearSchedule
from collections import namedtuple

import sys
import datetime
def print_now(cmd):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('%s %s' % (time_now, cmd))
    sys.stdout.flush()

# Arguments
parser = argparse.ArgumentParser(description='DQN Pytorch')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--log-dir', default='./agentLog',
                    help='directory to save agent logs (default: ./agentLog)')
parser.add_argument('--save-dir', default='./saved_model',
                    help='directory to save agent logs (default: ./saved_model)')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed (default: 1234)')
parser.add_argument('--save-interval', type=int, default=100,
                    help='save interval, one save per n updates (default: 100)')
parser.add_argument('--total-timestep', type=float, default=1e8,
                    help='total timestep (default: 1e8)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='num processes (default: 1)')
parser.add_argument('--gamma', type=int, default=0.99,
                    help='discount factor gamma (default 0.99)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default to use CUDA)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size in DQN (default: 32)')
parser.add_argument('--train-freq', type=int, default=4,
                    help='frequency in DQN training. Every 4 frames')
parser.add_argument('--target-update', type=int, default=32000,
                    help='frequency in target-network update. Every 1000 steps')
parser.add_argument('--memory-size', type=int, default=1000000,
                    help='memory size - 10,000 transitions')
parser.add_argument('--learning-starts', type=int, default=80000,
                    help='learning starts after - 80,000 transitions')
parser.add_argument('--num-lookahead', type=int, default=3,
                    help='look ahead step - 3 transitions')

parser.add_argument('--use-double-dqn',         action='store_true', default=False,
                    help='use-double-dqn')

parser.add_argument('--use-prioritized-buffer', action='store_true', default=False,
                    help='use-prioritized replay buffer')

parser.add_argument('--use-n-step', action='store_true', default=False,
                    help='use-prioritized replay buffer')

parser.add_argument('--use-duel', action='store_true', default=False,
                    help='use dueling architecture')

parser.add_argument('--use-noisy-net', action='store_true', default=False,
                    help='use dueling architecture')

parser.add_argument('--use-C51', action='store_true', default=False,
                    help='use categorical value distribution C51')

parser.add_argument('--use-QR-C51', action='store_true', default=False,
                    help='use categorical value distribution C51')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
GAMMA         = args.gamma
BATCH_SIZE    = args.batch_size
TRAIN_FREQ    = args.train_freq
TARGET_UPDATE         = args.target_update #

# Q-Learning Parameters
DOUBLE_Q_LEARNING  = args.use_double_dqn         #False
PRIORITIZED_MEMORY = args.use_prioritized_buffer #False
USE_N_STEP         = args.use_n_step
NUM_LOOKAHEAD      = args.num_lookahead
USE_DUEL           = args.use_duel
USE_NOISY_NET      = args.use_noisy_net
USE_C51            = args.use_C51
USE_QR_C51         = args.use_QR_C51
if USE_QR_C51:
    assert(USE_C51 is True)

if not USE_N_STEP:
    NUM_LOOKAHEAD = 1
# --------------------------------------------------- #
exploration_fraction    = 0.1
exploration_final_eps_1 = 0.1
exploration_final_eps_2 = 0.01
adam_lr =  6.25e-4 # 5e-5     if USE_QR_C51 else 
adam_eps = 1.5e-4 # 3.125e-4 if USE_QR_C51 else 
# --------------------------------------------------- #
# Booking Keeping
print_now('------- Begin DQN with --------')
print_now('Using Double DQN:                {}'.format(DOUBLE_Q_LEARNING))
print_now('Using Prioritized buffer:        {}'.format(PRIORITIZED_MEMORY))
print_now('Using N-step reward with N = {}:  {}'.format(NUM_LOOKAHEAD, USE_N_STEP))
print_now('Using Duel (advantage):          {}'.format(USE_DUEL))
print_now('Using Noisy Net:                 {}'.format(USE_NOISY_NET))
print_now('Using C51                        {}'.format(USE_C51))
print_now('Using Quantile Regression C51:   {}'.format(USE_QR_C51))
print_now('Adam learning rate: {}, eps: {}'.format(adam_lr, adam_eps))
print_now('Seed: {}'.format(args.seed))
print_now('------- -------------- --------')
print_now('Task: {}'.format(args.env_name))
time.sleep(0.1)


# Seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Importand - logging
try:
    print_now('Creating log directory at: %s' % (args.log_dir))
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)
    print_now('Reset log directory contents at: %s' % (args.log_dir))

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

# Network and Env following A2C-pytorch
device = torch.device("cuda" if args.cuda else "cpu")
print_now('Using device: {}'.format(device))
envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                    args.gamma, args.log_dir, args.add_timestep, device, False)

action_space = envs.action_space.n
if USE_C51:
    policy_net = C51(num_inputs=4, num_actions=action_space, 
                        use_duel=USE_DUEL, use_noisy_net=USE_NOISY_NET, use_qr_c51=USE_QR_C51).to(device)
    target_net = C51(num_inputs=4, num_actions=action_space, 
                        use_duel=USE_DUEL, use_noisy_net=USE_NOISY_NET, use_qr_c51=USE_QR_C51).to(device)
else:
    policy_net = DQN(num_inputs=4, num_actions=action_space, use_duel=USE_DUEL, use_noisy_net=USE_NOISY_NET).to(device)
    target_net = DQN(num_inputs=4, num_actions=action_space, use_duel=USE_DUEL, use_noisy_net=USE_NOISY_NET).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.eval()
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
optimizer = optim.Adam(policy_net.parameters(), lr=adam_lr, eps=adam_eps)
# -------------------------------------------------------------------######
if PRIORITIZED_MEMORY:
    memory    = PrioritizedReplayBuffer(args.memory_size, args.total_timestep, args.learning_starts)
else:
    memory    = ReplayMemory(args.memory_size)

nstep_buffer = []
def n_step_preprocess(st_0, action, st_1, reward, done):
    transition = Transition(st_0, action, st_1, reward)
    if done:
        # Clear out the buffer
        while len(nstep_buffer) > 1:
            n_step_reward = sum([nstep_buffer[i].reward.item()*(GAMMA**i) for i in range(len(nstep_buffer))])
            prev_transition = nstep_buffer.pop(0)
            temp_st0    = prev_transition.state
            temp_action = prev_transition.action
            temp_reward = torch.tensor([[n_step_reward]], dtype=torch.float)
            memory.push(temp_st0, temp_action, None, temp_reward)
        #
        n_step_reward   = sum([nstep_buffer[i].reward.item()*(GAMMA**i) for i in range(len(nstep_buffer))])
        prev_transition = nstep_buffer.pop(0)
        assert(len(nstep_buffer) == 0)
        return prev_transition.state, prev_transition.action, None, torch.tensor([[n_step_reward]], dtype=torch.float)

    elif len(nstep_buffer) < NUM_LOOKAHEAD - 1:
        nstep_buffer.append(transition)
        return None, None, None, None #st_0, action, st_1, reward
    else:
        nstep_buffer.append(transition)
        n_step_reward   = sum([nstep_buffer[i].reward.item()*(GAMMA**i) for i in range(NUM_LOOKAHEAD)])
        prev_transition = nstep_buffer.pop(0)
        # return prev_st0, prev_action, st_1, torch.tensor([[n_step_reward]], dtype=torch.float).to(device)
        assert(len(nstep_buffer) < NUM_LOOKAHEAD)
        return prev_transition.state, prev_transition.action, st_1, torch.tensor([[n_step_reward]], dtype=torch.float)
    #

# -------------------------------------------------------------------######
if USE_C51:
    C51_atoms =  51
    C51_vmax  =  10.0
    C51_vmin  = -10.0
    C51_support = torch.linspace(C51_vmin, C51_vmax, C51_atoms).view(1, 1, C51_atoms).to(device) # Shape  1 x 1 x 51
    C51_delta   = (C51_vmax - C51_vmin) / (C51_atoms - 1)

    if USE_QR_C51:
        QR_C51_atoms = C51_atoms
        QR_C51_quantile_weight = 1.0 / QR_C51_atoms
        QR_C51_cum_density = (2 * np.arange(QR_C51_atoms) + 1) / (2.0 * QR_C51_atoms)
        QR_C51_cum_density =  torch.tensor(QR_C51_cum_density, device=device, dtype=torch.float)

def next_distribution(non_final_next_states, batch_reward, non_final_mask):
    """
    This is for Quantile Regression C51
    """
    def get_action_argmax_next_Q_sa_QRC51(next_states):
        if DOUBLE_Q_LEARNING:
            next_dist = policy_net(next_states) * QR_C51_quantile_weight
        else:
            next_dist = target_net(next_states) * QR_C51_quantile_weight
        next_Q_sa = next_dist.sum(dim=2).max(1)[1]             # next_Q_sa is of size: [batch ] of action index 
        next_Q_sa = next_Q_sa.view(next_states.size(0), 1, 1)  # Make it to be size of [32 x 1 x 1]
        next_Q_sa = next_Q_sa.expand(-1, -1, QR_C51_atoms)        # Expand to be [32 x 1 x 51], one action, expand to support
        return next_Q_sa

    with torch.no_grad():
        quantiles_next = torch.zeros((BATCH_SIZE, QR_C51_atoms), device=device, dtype=torch.float)
        max_next_action                = get_action_argmax_next_Q_sa_QRC51(non_final_next_states)
        if USE_NOISY_NET:
            target_net.sample_noise()        
        quantiles_next[non_final_mask] = target_net(non_final_next_states).gather(1, max_next_action).squeeze(1) 
        # output should change from [32 x 1 x 51] --> [32 x 51]
        # batch_reward should be of size [32 x 1]
        quantiles_next = batch_reward + (GAMMA**NUM_LOOKAHEAD) * quantiles_next
    return quantiles_next



def project_distribution(batch_state, batch_action, non_final_next_states, batch_reward, non_final_mask):
    """
    This is for orignal C51, with KL-divergence.
    """
    def get_action_argmax_next_Q_sa(next_states):
        if DOUBLE_Q_LEARNING:
            next_dist = policy_net(next_states) * C51_support      # Next_Distribution is of size: [batch x action x atoms]  
        else:
            next_dist = target_net(next_states) * C51_support      # Next_Distribution is of size: [batch x action x atoms]  
        next_Q_sa = next_dist.sum(dim=2).max(1)[1]             # next_Q_sa is of size: [batch ] of action index 
        next_Q_sa = next_Q_sa.view(next_states.size(0), 1, 1)  # Make it to be size of [32 x 1 x 1]
        next_Q_sa = next_Q_sa.expand(-1, -1, C51_atoms)        # Expand to be [32 x 1 x 51], one action, expand to support
        return next_Q_sa

    with torch.no_grad():
        max_next_dist = torch.zeros((BATCH_SIZE, 1, C51_atoms), device=device, dtype=torch.float)
        max_next_dist += 1.0 / C51_atoms
        #
        max_next_action               = get_action_argmax_next_Q_sa(non_final_next_states)
        if USE_NOISY_NET:
            target_net.sample_noise()
        max_next_dist[non_final_mask] = target_net(non_final_next_states).gather(1, max_next_action)
        max_next_dist = max_next_dist.squeeze()
        #
        # Mapping
        Tz = batch_reward.view(-1, 1) + (GAMMA**NUM_LOOKAHEAD) * C51_support.view(1, -1) * non_final_mask.to(torch.float).view(-1, 1)
        Tz = Tz.clamp(C51_vmin, C51_vmax)
        C51_b = (Tz - C51_vmin) / C51_delta
        C51_L = C51_b.floor().to(torch.int64)
        C51_U = C51_b.ceil().to(torch.int64)
        C51_L[ (C51_U > 0)               * (C51_L == C51_U)] -= 1
        C51_U[ (C51_L < (C51_atoms - 1)) * (C51_L == C51_U)] += 1
        offset = torch.linspace(0, (BATCH_SIZE - 1) * C51_atoms, BATCH_SIZE)
        offset = offset.unsqueeze(dim=1) 
        offset = offset.expand(BATCH_SIZE, C51_atoms).to(batch_action) # I believe this is to(device)

        # I believe this is analogous to torch.new_zeros()
        m = batch_state.new_zeros(BATCH_SIZE, C51_atoms) # Returns a Tensor of size size filled with 0. same dtype
        m.view(-1).index_add_(0, (C51_L + offset).view(-1), (max_next_dist * (C51_U.float() - C51_b)).view(-1))
        m.view(-1).index_add_(0, (C51_U + offset).view(-1), (max_next_dist * (C51_b - C51_L.float())).view(-1))
    return m
# -------------------------------------------------------------------######



# -------------------------------------------------------------------######
# Two stage epsilon decay following https://blog.openai.com/openai-baselines-dqn/
# But this is basically like expoenntial decay
eps_schedule1 = LinearSchedule(schedule_timesteps=int(1e6),  # first 1 million
                              initial_p=1.0,
                              final_p  =exploration_final_eps_1)

eps_schedule2 = LinearSchedule(schedule_timesteps=int(25e6), # next 24 million
                              initial_p=exploration_final_eps_1,
                              final_p  =exploration_final_eps_2)

steps_done = 0
def select_action(state, action_space):
    global steps_done
    # eps_threshold = EPS_END + (EPS_STRAT-EPS_END) * math.exp(-1*steps_done / EPS_DECAY)
    eps_threshold = eps_schedule1.value(steps_done) if steps_done <= 1e6 else eps_schedule2.value(steps_done)
    steps_done += 1
    if USE_NOISY_NET or random.random() > eps_threshold:
        with torch.no_grad():
            if USE_QR_C51:
                if USE_NOISY_NET:
                    policy_net.sample_noise()
                y = policy_net(state)
                y = y * QR_C51_quantile_weight
                y = y.sum(dim=2).max(1)
                action = y[1].view(1, 1)

            elif USE_C51:
                if USE_NOISY_NET:
                    policy_net.sample_noise()                
                y = policy_net(state)
                y = y * C51_support
                y = y.sum(dim=2).max(1)
                action = y[1].view(1, 1)
            else:
                if USE_NOISY_NET:
                    policy_net.sample_noise()
                y = policy_net(state)
                y = y.max(1) # (tensor([0.2177], grad_fn=<MaxBackward0>), tensor([0]))
                action = y[1].view(1, 1)
    else:
        action = torch.tensor([[random.randrange(action_space)]], device=device, dtype=torch.long)
    return action

# optimize
def optimize_model():
    #
    # F.smooth_l1_loss(x, x.zero())
    def huber_loss(x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    # print_now('in optimize_model, device = {}'.format(device))
    if PRIORITIZED_MEMORY:
        transitions, batch_index, batch_weight_IS = memory.sample(BATCH_SIZE)
    else:
        transitions = memory.sample(BATCH_SIZE)

    batch       = Transition(*zip(*transitions))
    non_final             = tuple(map(lambda s: s is not None,  batch.next_state))
    non_final_mask        = torch.tensor(non_final, device=device, dtype=torch.uint8)
    sanity_check          = [s for s in batch.next_state if s is not None]
    if len(sanity_check) == 0:
        return None, None, None
    non_final_next_states = torch.cat(sanity_check).to(device)
    #
    state_batch           = torch.cat(batch.state).to(device)
    action_batch          = torch.cat(batch.action).to(device) # this is of shape [32 x 1]
    reward_batch          = torch.cat(batch.reward).to(device)
    batch_weight_IS       = torch.tensor(batch_weight_IS).to(device) # [32,]
    #
    if USE_QR_C51:
        QR_C51_action = action_batch.unsqueeze(dim=-1).expand(-1, -1, QR_C51_atoms)
        QR_C51_reward = reward_batch.view(-1, 1) # [32 x 1]
        #
        if USE_NOISY_NET:
            policy_net.sample_noise()        
        y = policy_net(state_batch)
        quantiles     = y.gather(1, QR_C51_action)      # [32 x 1 x 51]
        quantiles     = quantiles.squeeze(1) # [32 x 51]
        #
        quantiles_next = next_distribution(non_final_next_states, QR_C51_reward, non_final_mask) # [32, 51]
        #
        #              [51 x 32 x 1 ]                [1, 32, 51]
        diff = quantiles_next.t().unsqueeze(-1) - quantiles.unsqueeze(0) # diff is of shape [51, 32 51]
        loss = huber_loss(diff) * torch.abs( QR_C51_cum_density.view(1, -1) - (diff < 0).to(torch.float) )

        # loss is now of shape [51, 32, 51]
        loss = loss.transpose(0,1) # loss is now of shape [32, 51, 51]
        if PRIORITIZED_MEMORY:
            loss = loss * batch_weight_IS.view(BATCH_SIZE, 1, 1)            
        loss = loss.mean(1).sum(-1)
        loss_PER = loss.detach().squeeze().abs().cpu().numpy()    
        loss = loss.mean()
        #
        ds = y.detach() * QR_C51_quantile_weight
        Q_sa = ds.sum(dim=2).gather(1, action_batch)        
    elif USE_C51:
        #                           [32 x 1 x 1]           [32 x 1 x 51]
        C51_action   = action_batch.unsqueeze(dim=-1).expand(-1, -1, C51_atoms)
        C51_reward   = reward_batch.view(-1, 1, 1) # [32 x 1 x 1]
        #                            [32 x 1 x 51]         --->           [32 x 51]
        if USE_NOISY_NET:
            policy_net.sample_noise()
        y = policy_net(state_batch)
        current_dist = y.gather(1, C51_action).squeeze()
        target_prob  = project_distribution(state_batch, C51_action, non_final_next_states, C51_reward, non_final_mask) # torch.Size([32, 51])
        loss = -(target_prob * current_dist.log()).sum(-1) # KL Divergence
        loss_PER = loss.detach().squeeze().abs().cpu().numpy()
        if PRIORITIZED_MEMORY:
            loss = loss * batch_weight_IS.view(BATCH_SIZE, 1)
        loss = loss.mean()
        #
        ds = y.detach() * C51_support
        Q_sa = ds.sum(dim=2).gather(1, action_batch)
    else:
    #   # Normal DQN. Minimize expected TD error ------------------------######
        Q_sa          = policy_net(state_batch).gather(1, action_batch)
        next_Q_sa     = torch.zeros((BATCH_SIZE, 1), device=device)
        if DOUBLE_Q_LEARNING:
            # Double DQN, getting action from policy net. 
            # See https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
            with torch.no_grad():
                target_Q_sa             = target_net(non_final_next_states)
                action_from_policy_Q_sa = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)  # max of the first dimension --> tuple(val, index). 
                Q_sa_double_DQN = target_Q_sa.gather(1, action_from_policy_Q_sa)                    # We use the action index from policy net
                next_Q_sa[non_final_mask] = Q_sa_double_DQN
        else:
            # Vanilla DQN, getting action from target_net
            with torch.no_grad():
                target_Q_sa = target_net(non_final_next_states)
                Q_sa_DQN    = target_Q_sa.max(1)[0].unsqueeze(1)
                next_Q_sa[non_final_mask] = Q_sa_DQN

        Expected_Q_sa = reward_batch + ((GAMMA**NUM_LOOKAHEAD) * next_Q_sa)
        #
        if PRIORITIZED_MEMORY:
            diff = Q_sa - Expected_Q_sa
            loss = huber_loss(diff).squeeze() * batch_weight_IS
        else:
            loss = F.smooth_l1_loss(Q_sa, Expected_Q_sa)
    # -------------------------------------------------------------------######
    if PRIORITIZED_MEMORY:
        if USE_C51 or USE_QR_C51:
            memory.update_priority_on_tree(batch_index, loss_PER)
        else:
            TD_error = Q_sa.detach() - Expected_Q_sa.detach()
            TD_error = TD_error.cpu().numpy().squeeze()
            abs_TD_error = abs(TD_error)
            memory.update_priority_on_tree(batch_index, abs_TD_error)
    # -------------------------------------------------------------------######
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # -------------------------------------------------------------------######
    Qval = Q_sa.cpu().detach().numpy().squeeze()
    return loss, np.mean(Qval), np.mean(reward_batch.cpu().numpy().squeeze())
    #

def save_model():
    save_path = os.path.join(args.save_dir)
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    #
    # Convert model to CPU
    save_model = policy_net
    if args.cuda:
        save_model = copy.deepcopy(policy_net).cpu()
    # save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None)]
    torch.save(save_model, 
               os.path.join(save_path, "%s.pt"%(args.env_name)))
    gc.collect()

# main
def main():
    global steps_done
    torch.set_num_threads(1)
    loss = None
    Q_sa = None
    batch_reward_mean = None
    update_count = 0
    action_history  = deque(maxlen=1000)
    episode_rewards = deque(maxlen=100)
    # -------------------------------------------------------------------######
    # if PRIORITIZED_MEMORY:
    #     state = PER_pre_fill_memory(envs) # reset would be called inside
    # else:
    #     state = envs.reset()
    state = envs.reset()
    # -------------------------------------------------------------------######    
    start = time.time()
    for t in range(int(args.total_timestep)):
        action = select_action(state, action_space)
        action_history.append(action.item())
        st_0 = copy.deepcopy(state)      # IMPORTANT. Make a deep copy as state will be come next_state AUTOMATICALLY

        next_state, reward, done, info = envs.step(action)
        st_1 = copy.deepcopy(next_state) # Just to re-iterate the importance, that's all
        if 'episode' in info[0].keys():
            episode_rewards.append(info[0]['episode']['r'])
        # We only ensure one environment here
        # -------------------------------------------------------------------######    
        if USE_N_STEP:
            st_0, action, st_1, reward = n_step_preprocess(st_0, action, st_1, reward, done[0])
        # -------------------------------------------------------------------######        
        if done[0]:
            memory.push(st_0, action, None, reward)
        elif st_0 is not None:
            memory.push(st_0, action, st_1, reward)
        state = next_state
        #
        if t > args.learning_starts and t % TRAIN_FREQ == 0:
            update_count += 1
            loss, Q_sa, batch_reward_mean = optimize_model()
            if t % args.save_interval == 0:
                save_model()
        #
        if t > args.learning_starts and t % TARGET_UPDATE == 0:
            print_now('Updated target network at %d' % (t))
            target_net.load_state_dict(policy_net.state_dict())

        # Book Keeping
        end = time.time()
        eps_threshold = eps_schedule1.value(steps_done) if steps_done <= 1e6 else eps_schedule2.value(steps_done)

        if t%500 == 0 and len(episode_rewards) > 0:
            print_now('Upd {} timestep {} FPS {} - last {} ep rew: mean : {:.1f} min/max: {:.1f}/{:.1f} action_std: {:.3f} eps_val: {:.4f} loss: {:.4f} Qval {:.2f} Nrew: {:.2f}'.format(
                update_count, t,
                int(t / (end-start)),
                len(episode_rewards), np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards),
                np.std(action_history), eps_threshold, 
                loss.item()       if loss              else -9999, 
                Q_sa              if Q_sa              else 0,
                batch_reward_mean if batch_reward_mean else 0
                ))
        elif len(episode_rewards) == 0:
            print_now('Upd {}, timestep {}, FPS {}'.format(
                update_count, t,
                int(t / (end-start)),
                len(episode_rewards), -1, -1, -1
                ))            
        #
    #
#
if __name__ == "__main__":
    main()


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


""" A2C specific arguments """
#import algo
#from arguments import get_args

# from envs  import make_vec_envs
# from model import Policy
# from storage import RolloutStorage
# from utils import get_vec_normalize

# DQN specific arguments
from DQN_network import DQN
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
parser.add_argument('--target-update', type=int, default=1000,
                    help='frequency in target-network update. Every 1000 steps')
parser.add_argument('--memory-size', type=int, default=10000,
                    help='memory size - 10,000 transitions')
parser.add_argument('--learning-starts', type=int, default=10000,
                    help='learning starts after - 10,000 transitions')

parser.add_argument('--use-double-dqn',         action='store_true', default=False,
                    help='use-double-dqn')

parser.add_argument('--use-prioritized-buffer', action='store_true', default=False,
                    help='use-prioritized replay buffer')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
GAMMA         = args.gamma
BATCH_SIZE    = args.batch_size
TRAIN_FREQ    = args.train_freq
TARGET_UPDATE         = args.target_update #

exploration_fraction    = 0.1
exploration_final_eps_1 = 0.1
exploration_final_eps_2 = 0.01
lr = 1e-4
# Q-Learning Parameters
DOUBLE_Q_LEARNING  = args.use_double_dqn         #False
PRIORITIZED_MEMORY = args.use_prioritized_buffer #False

# Booking Keeping
print_now('------- Begin DQN with --------')
print_now('Using Double DQN: {}'.format(DOUBLE_Q_LEARNING))
print_now('Using Prioritized buffer: {}'.format(PRIORITIZED_MEMORY))
print_now('Seed: {}'.format(args.seed))
print_now('------- -------------- --------')
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
policy_net = DQN(num_inputs=4, num_actions=action_space).to(device)
target_net = DQN(num_inputs=4, num_actions=action_space).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
# -------------------------------------------------------------------######
if PRIORITIZED_MEMORY:
    memory    = PrioritizedReplayBuffer(args.memory_size)
else:
    memory    = ReplayMemory(args.memory_size)
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
    if random.random() > eps_threshold:
        with torch.no_grad():
            y = policy_net(state)
            y = y.max(1) # (tensor([0.2177], grad_fn=<MaxBackward0>), tensor([0]))
            action = y[1].view(1, 1)
    else:
        action = torch.tensor([[random.randrange(action_space)]], device=device, dtype=torch.long)
    return action

# optimize
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
def optimize_model():
    # print_now('in optimize_model, device = {}'.format(device))
    if PRIORITIZED_MEMORY:
        transitions, batch_index, batch_weight_IS = memory.sample(BATCH_SIZE)
    else:
        transitions = memory.sample(BATCH_SIZE)

    batch       = Transition(*zip(*transitions))
    non_final             = tuple(map(lambda s: s is not None,  batch.next_state))
    non_final_mask        = torch.tensor(non_final, device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    #
    state_batch           = torch.cat(batch.state)
    action_batch          = torch.cat(batch.action)
    reward_batch          = torch.cat(batch.reward).to(device)
    #
    Q_sa          = policy_net(state_batch).gather(1, action_batch)
    next_Q_sa     = torch.zeros((BATCH_SIZE, 1), device=device)
    if DOUBLE_Q_LEARNING:
        # Double DQN, getting action from policy net. 
        # See https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
        with torch.no_grad():
            target_Q_sa             = target_net(non_final_next_states)
            action_from_policy_Q_sa = policy_net(non_final_next_states).max(1)[1]  # max of the first dimension --> tuple(val, index). 
            Q_sa_double_DQN = target_Q_sa[0][action_from_policy_Q_sa].unsqueeze(1) # We use the action index from policy net
            next_Q_sa[non_final_mask] = Q_sa_double_DQN
    else:
        # Vanilla DQN, getting action from target_net
        with torch.no_grad():
            target_Q_sa = target_net(non_final_next_states)
            Q_sa_DQN    = target_Q_sa.max(1)[0].unsqueeze(1)
            next_Q_sa[non_final_mask] = Q_sa_DQN

    Expected_Q_sa = reward_batch + (GAMMA * next_Q_sa)
    #
    loss = F.smooth_l1_loss(Q_sa, Expected_Q_sa)
    # -------------------------------------------------------------------######
    if PRIORITIZED_MEMORY:
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

def PER_pre_fill_memory(envs):
    """
    Pre-filling the memory buffer if we are doing Prioritized Experience Replay 
    """
    state = envs.reset()
    print_now('Begin to pre-fill [Prioritized Experience Replay Memory]')
    for j in range(args.memory_size):
        action = torch.tensor([[random.randrange(action_space)]], device=device, dtype=torch.long)
        st_0 = copy.deepcopy(state)      # IMPORTANT. Make a deep copy as state will be come next_state AUTOMATICALLY
        next_state, reward, done, info = envs.step(action)
        st_1 = copy.deepcopy(next_state)
        # We only ensure one environment here
        if done[0]:
            memory.push(st_0, action, None, reward)
        else:
            memory.push(st_0, action, st_1, reward)      
        state = next_state  
        print_now('Pre-filling Replay Memory %d / %d -- action: %d' % (j+1, args.memory_size, action.item()))
    return state
    #

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
    if PRIORITIZED_MEMORY:
        state = PER_pre_fill_memory(envs) # reset would be called inside
    else:
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
        if done[0]:
            memory.push(st_0, action, None, reward)
        else:
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


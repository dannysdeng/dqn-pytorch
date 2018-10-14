import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

import sys
import datetime
def print_now(cmd):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('%s %s' % (time_now, cmd))
    sys.stdout.flush()

# Make sure this is an atari environment (best if it is MuJoCo Compatible)
def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets):
    assert(log_dir is not None)
    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        
        if add_timestep:
            if len(obs_shape) == 1 and str(env).find(TimeLimt) > -1:
                print_now('Adding timestep wrapper to env')
                env = AddTimestep(env)

        env = bench.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)
        if is_atari:
            env = wrap_deepmind(env)
        # If the input is of shape (W, H, 3), wrap for PyTorch (N, 3, W, H)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)
        return env 
    return _thunk

###
# Vectorizer to give [4x84x84 x num_processes]
###
def make_vec_envs(env_name, seed, num_processes, gamma, log_dir,
                  add_timestep, device, allow_early_resets, num_frame_stack=None):
    envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets) for i in range(num_processes)]
    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    # 
    # This is for MuJoCo Maybe?
    if len(envs.observation_space.shape) == 1:
        print_now('Performning VecNormalize as observation_space is of shape 1')
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)
    #
    envs = VecPyTorch(envs, device)
    if num_frame_stack is not None:
        # If there is some pre-defined framestack:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        print_now('Using default 4-frame stack for image-based envs')
        envs = VecPyTorchFrameStack(envs, 4, device)
    # 
    return envs

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """ Return only every 'skip'-th frame """
        super(VecPyTorch, self).__init__(venv)
        self.device = device
    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs
    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)
    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs    = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float() # N --> N x 1
        return obs, reward, done, info

class VecPyTorchFrameStack(VecEnvWrapper):
    """ OpenAI-baseline style framestack """
    def __init__(self, venv, nstack, device=None):
        self.venv   = venv
        self.nstack = nstack
        wrapped_ob_space = venv.observation_space # should be 1 x 84 x 84
        self.shape_dim0  = wrapped_ob_space.shape[0] # shape_dim0 is 1

        # wrapped_ob_space.low is ZERO matrix of size 1 x 84 x 84, we make it 4 x 84 x 84 now
        # wrapped_ob_space.high is 255-matrix of size 1 x 84 x 84, we make it 4 x 84 x 84 now
        low  = np.repeat(wrapped_ob_space.low,  self.nstack, axis=0) 
        high = np.repeat(wrapped_ob_space.high, self.nstack, axis=0) 

        if device is None:
            device = torch.device('cpu')
        new_shape_tuple = (venv.num_envs, ) + low.shape # num_processes x 4 x 84 x 84 
        self.stacked_obs = torch.zeros(new_shape_tuple).to(device)

        observation_space = gym.spaces.Box( 
            low=low, high=high, dtype=venv.observation_space.dtype)

        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # This is stacking 4 frames together
        # self.stacked_obs[:, :-1] is everything (first 3) except the last one, self.stacked_obs[:, -1:] is everything (last 3) except the first one
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:] # essentially pops the first 1 out
        for i, done in enumerate(dones):
            if done:
                self.stacked_obs[i] = 0
        #
        # self.stacked_obs[:, -1:] = obs
        self.stacked_obs[:, -self.shape_dim0:] = obs # put the new observation at the last 1 position
        return self.stacked_obs, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        # Zero-out everything in the stacked env
        self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs # put the first state (new observation) at the last 1 position
        return self.stacked_obs

    def close(self):
        self.venv.close()


## Helper Wraper:
class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))

class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0],
                [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        # Observation is of type Tensor
        return observation.transpose(2, 0, 1)


class VecNormalize(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False








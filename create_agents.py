import torch
from networks.dqn_lunarlander import DQN
import gymnasium as gym
from networks.ddpg_fetchrobot import Agent as DDPG
from copy import deepcopy as dc

"""
This file contains functions to create learning agents for each domain.
Each function loads a pre-trained model from a file and returns an agent instance.
"""


def create_lunarlander_agent(file):
    agent = DQN(
        n_observations = 11,
        n_actions = 4,
        batch_size = 128,
        lr = 0.05,
        gamma = 0.99,
        mem_size = 250000,
        learn_step = 5,
        tau = 0.005,
        )
    
    checkpoint_p = torch.load(file, map_location="cpu")
    agent.policy_net.load_state_dict(checkpoint_p['model_state_dict'])

    return agent

def create_robot_agent(file):
    memory_size = 7e+5 // 50
    batch_size = 256
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.98
    tau = 0.05
    k_future = 4
    ENV = gym.make('FetchPickAndPlace-v2', max_episode_steps=650)
    n_observations = ENV.observation_space.spaces["observation"].shape
    n_actions = ENV.action_space.shape[0]
    n_goals = ENV.observation_space.spaces["desired_goal"].shape[0]
    action_bounds = [ENV.action_space.low[0], ENV.action_space.high[0]]

    agent = DDPG(n_states=n_observations,
              n_actions=n_actions,
              n_goals=n_goals,
              action_bounds=action_bounds,
              capacity=memory_size,
              action_size=n_actions,
              batch_size=batch_size,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              gamma=gamma,
              tau=tau,
              k_future=k_future,
              env=dc(ENV))
    
    checkpoint = torch.load(file)
    actor_state_dict = checkpoint["actor_state_dict"]
    agent.actor.load_state_dict(actor_state_dict)
    state_normalizer_mean = checkpoint["state_normalizer_mean"]
    agent.state_normalizer.mean = state_normalizer_mean
    state_normalizer_std = checkpoint["state_normalizer_std"]
    agent.state_normalizer.std = state_normalizer_std
    goal_normalizer_mean = checkpoint["goal_normalizer_mean"]
    agent.goal_normalizer.mean = goal_normalizer_mean
    goal_normalizer_std = checkpoint["goal_normalizer_std"]
    agent.goal_normalizer.std = goal_normalizer_std

    return agent

def create_flappy_agent(file):
    agent = DQN(
        n_observations = 12,
        n_actions = 2,
        batch_size = 128,
        lr = 0.05,
        gamma = 0.99,
        mem_size = 250000,
        learn_step = 5,
        tau = 0.005,
        )
    
    checkpoint_p = torch.load(file, map_location="cpu")
    agent.policy_net.load_state_dict(checkpoint_p['model_state_dict'])

    return agent

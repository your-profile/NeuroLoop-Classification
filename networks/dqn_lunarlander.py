import pandas as pd
import argparse
import torch
from games.lunar_lander import LunarLander
from games.flappy_bird import FlappyBirdEnv
import os
import torch.nn as nn
import numpy as np
import pygame
import pickle
import datetime
import torch.optim as optim
from collections import deque
import time
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Q NETWORK ARCHITECTURE
class DeepQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DeepQNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.fc(x)
    
class DQN():
    def __init__(self, n_observations, n_actions, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=5, tau=1e-3):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau

        self.policy_net = DeepQNetwork(n_observations, n_actions).to(device)
        self.target_net = DeepQNetwork(n_observations, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # REplay Buffer
        # self.memory = PrioritizedReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0

    def chooseAction(self, state, epsilon, play=None):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # print(state)

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        if play: return action_values

        # epsilon-greedy
        if np.random.random() < epsilon:
            action = np.random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())

        return action, action_values
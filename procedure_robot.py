import pandas as pd
import argparse
import torch
import os
import torch.nn as nn
import numpy as np
import pygame
import pickle
import datetime
import torch.optim as optim
from torch.optim import Adam
from copy import deepcopy as dc
import time
import torch.nn.functional as F
from torch import from_numpy
import gymnasium as gym

START_WAIT = 2000 #Time participants wait at the Enter Screen
EPISODE_WAIT = 1500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

pygame.init()

# Initialize the joystick module
pygame.joystick.init()

# Get the number of joysticks
joystick_count = pygame.joystick.get_count()
# assert(joystick_count>0)

# print(f"Number of joysticks connected: {joystick_count}")

# Loop through and list all connected joysticks
for i in range(joystick_count):
    joystick = pygame.joystick.Joystick(i)
    joystick.init()

# Policy file is actually the policy
parser.add_argument("-c", "--condition", type=int, required=True, help="Game Condition")
parser.add_argument("-id", "--PID", type=str, required=True, help="Participant ID")
parser.add_argument("-n", "--demos", type=int, required=True, help="Number of Demonstrations")

if not os.path.exists('data/human-play-demo/'):
            os.makedirs('data/human-play-demo/')
if not os.path.exists('data/human-watch-demo/'):
            os.makedirs('data/human-watch-demo/')

args = parser.parse_args()

def environment(condition):
    if condition == 0:
        return gym.make('FetchPickAndPlace-v2', render_mode = "human", max_episode_steps=75)
    if condition == 1:
        return gym.make('FetchPickAndPlace-v2', render_mode = "human", max_episode_steps=650)

    if condition == 2:
        return gym.make('FetchPush-v2', render_mode = "human", max_episode_steps=75)
    
    if condition == 3:
        return gym.make('FetchPush-v2', render_mode = "human", max_episode_steps=650)

CONDITION = args.condition
PARTICIPANT_ID = args.PID
CONDITION_MAP = {
    0: ['w', environment(CONDITION), "PickPlaceRobot", (25,3,4), "policies/RobotPolicies/FetchPickAndPlace1.pth"],
    1: ['p', environment(CONDITION), "PickPlaceRobot", (25,3,4), "FetchPickAndPlace.pth"],
    2: ['w', environment(CONDITION), "PushRobot", (2,12), "RobotPolicy"],
    3: ['p', environment(CONDITION), "PushRobot", (2,12), "RobotPolicy"]
}

TASK, ENV, ENVIRONMENT_NAME, SPACE, EXPERT_FILENAME = CONDITION_MAP[args.condition]
NUM_DEMOS = args.demos

POLICY = [EXPERT_FILENAME]

n_actions, n_goals, n_observations = SPACE
env_params = {"action_max": 1.0, "obs": n_observations, "goal": n_goals, "action": n_actions}

def main():
    if TASK == 'w':
        watch(POLICY, condition=CONDITION)
    elif TASK == 'p':
        play(condition = CONDITION)
    else:
         Exception("wrong Condition Input")

def transition():
    rand = np.random.rand()

    if rand < 0.30:
        return np.random.randint(3,6)
    
    return 0
    

def watch(policies, condition:int):
    first_time = time.time()
    env = ENV
    env.reset()
    subopt_count = 0
    seed = np.random.randint(0, 2000)

    agent = create_agent(EXPERT_FILENAME)
    new_seed = seed

    demonstration_dict = {}

    #for each demonstration desired
    for i_episode in range(0, NUM_DEMOS):

        timestamps, timestamp_datetime, rewards_list, states_list, chosen_actions, optimal_actions = [], [], [], [], [], [] #timestamps
        
        state_dict, _ = env.reset(seed=new_seed) #reset
        num = np.random.rand()

        state = state_dict["observation"]
        desired_goal = state_dict["desired_goal"]
        achieved_goal = state_dict["achieved_goal"]

        timestamps.append(time.time())
        timestamp_datetime.append(pd.to_datetime(datetime.datetime.now().timestamp() * 1000, unit='ms'))
        states_list.append(state)
        rewards_list.append(0)
        optimal_actions.append(None)
        chosen_actions.append(None)

        done = False
        win, t = 0, 0

        if i_episode == 0:
                pygame.time.wait(START_WAIT)

        random_goal = np.array([np.random.uniform(1.0, 1.49), np.random.uniform(0.4, 1.1), np.random.uniform(0.4, 0.9)])

        while not done:
            time.sleep(0.1)
            env.render()

            #action selected and saved
            action = agent.choose_action(state, desired_goal, train_mode=False)
            # print(desired_goal)            
            current_time_float = time.time()
            current_time = pd.to_datetime(datetime.datetime.now().timestamp() * 1000, unit='ms')

            if subopt_count > 0 and t>np.random.randint(0, 5):
                if num < 1:
                    chosen_action = action + (np.random.randint(2, 6)*np.random.randn(4))
                else:
                    chosen_action = agent.choose_action(state, random_goal, train_mode=False)
            else:
                chosen_action = action

            chosen_actions.append(chosen_action)
            optimal_actions.append(action)

            next_state_dict, reward, term, done, _ = env.step(chosen_action)
 

            desired_goal = next_state_dict["desired_goal"]
            achieved_goal = next_state_dict["achieved_goal"]

            state = next_state_dict["observation"]

            state_dict = next_state_dict


            states_list.append(state_dict["observation"])

            rewards_list.append(reward)

            #timestamp update
            timestamps.append(current_time_float)
            timestamp_datetime.append(current_time)

            t += 1
            if reward > -1.0:
                win += 1

            # print(np.round(desired_goal, 2))
            if done or win > 10:
                pygame.time.wait(EPISODE_WAIT)
                break
        
        subopt_count -= 1

        if subopt_count < 0 and i_episode > 2:
            subopt_count = transition()
        
        final_episode_optimal_actions = optimal_actions
        final_episode_chosen_actions = chosen_actions
        final_episode_rewards = rewards_list
        final_episode_states = states_list
        final_episode_steps = t + 1


        assert(len(timestamps) == len(final_episode_chosen_actions) == len(final_episode_rewards)== (len(final_episode_states)))
        episode = save_demonstration(environment_name=ENVIRONMENT_NAME, steps = final_episode_steps, timestamp_datetime=timestamp_datetime, optimal_actions=final_episode_optimal_actions, chosen_actions=final_episode_chosen_actions, states=final_episode_states, timestamps=timestamps, rewards=final_episode_rewards, seed=new_seed, condition=condition)
        demonstration_dict[i_episode] = episode

        # print("Episode {} Collected Successfully".format(i_episode))

        demonstration_dict["NumberOfDemos"] = i_episode + 1
        new_seed += 1


    # Convert to a hashmap
    with open('./data/human-watch-demo/{}.pickle'.format("{}_WatchedDemonstration{}_C{}".format(PARTICIPANT_ID, ENVIRONMENT_NAME, condition)), 'wb') as handle:
        pickle.dump(demonstration_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    last_time = time.time()

    print(last_time-first_time, " seconds long")
    

def play(condition:int):
    
    first_time = time.time()

    env = ENV
    subopt_count = 0
    seed = np.random.randint(0, 2000)
    env.reset()


    agent = create_agent(EXPERT_FILENAME)
    new_seed = seed

    demonstration_dict = {}
    #for each demonstration desired

    for i_episode in range(0, NUM_DEMOS+1):
        

        timestamps, timestamp_datetime, rewards_list, states_list, chosen_actions, optimal_actions = [], [], [], [], [], [] #timestamps
        
        state_dict, _ = env.reset(seed=new_seed) #reset
        state = state_dict["observation"]
        desired_goal = state_dict["desired_goal"]
        achieved_goal = state_dict["achieved_goal"]

        

        timestamps.append(time.time())
        timestamp_datetime.append(pd.to_datetime(datetime.datetime.now().timestamp() * 1000, unit='ms'))
        states_list.append(state)
        rewards_list.append(0)
        optimal_actions.append(None)
        chosen_actions.append(None)

        done = False
        t, win = 0,0

        env.render()

        if i_episode == 0:
            pygame.time.wait(START_WAIT)

        while not done:
            time.sleep(0.05)
            env.render()

            #action selected and saved
            action = human_play()
            agent_action = agent.choose_action(state, desired_goal, train_mode=False)
            
            current_time_float = time.time()
            current_time = pd.to_datetime(datetime.datetime.now().timestamp() * 1000, unit='ms')

            chosen_actions.append(action)
            optimal_actions.append(agent_action)

            next_state_dict, reward, term, done, _ = env.step(action)

            desired_goal = next_state_dict["desired_goal"]
            achieved_goal = next_state_dict["achieved_goal"]
            state = next_state_dict["observation"]

            state_dict = next_state_dict
            states_list.append(state)

            rewards_list.append(reward)

            # #timestamp update
            timestamps.append(current_time_float)
            timestamp_datetime.append(current_time)

            t += 1

            if reward > -1.0:
                win += 1

            if done or win > 10:
                pygame.time.wait(EPISODE_WAIT)
                break
        
        final_episode_optimal_actions = optimal_actions
        final_episode_chosen_actions = chosen_actions
        final_episode_rewards = rewards_list
        final_episode_states = states_list
        final_episode_steps = t + 1

        if i_episode >= 0:
            assert(len(timestamps) == len(final_episode_chosen_actions) == len(final_episode_rewards)== (len(final_episode_states)))
            episode = save_demonstration(environment_name=ENVIRONMENT_NAME, steps = final_episode_steps, timestamp_datetime=timestamp_datetime, optimal_actions=final_episode_optimal_actions, chosen_actions=final_episode_chosen_actions, states=final_episode_states, timestamps=timestamps, rewards=final_episode_rewards, seed=new_seed, condition=condition)
            demonstration_dict[i_episode] = episode

            # print("Episode {} Collected Successfully".format(i_episode))

        new_seed += 1

        demonstration_dict["NumberOfDemos"] = i_episode + 1
    # Convert to a hashmap
    with open('./data/human-play-demo/{}.pickle'.format("{}_PlayedDemonstration{}_C{}".format(PARTICIPANT_ID, ENVIRONMENT_NAME, condition)), 'wb') as handle:
        pickle.dump(demonstration_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    last_time = time.time()
    print(last_time-first_time, " seconds long")



def cross_entropy(y_pred, y_true):
 
    y_pred = softmax(y_pred)
    loss = 0
     
    for i in range(len(y_pred)):
        loss = loss + (-1 * y_true[i]*np.log(y_pred[i]))
 
    return loss

def softmax(values):
 
    exp_values = np.exp(values, dtype=np.float128)
    exp_values_sum = np.sum(exp_values)
 
    vals = np.asarray(exp_values/exp_values_sum, dtype=np.float64)

    return vals

def KLDivergence(P, Q):
    from scipy.special import rel_entr

    return sum(rel_entr(P, Q))
    
def create_agent(file):
    memory_size = 7e+5 // 50
    batch_size = 256
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.98
    tau = 0.05
    k_future = 4
    n_observations = ENV.observation_space.spaces["observation"].shape
    n_actions = ENV.action_space.shape[0]
    n_goals = ENV.observation_space.spaces["desired_goal"].shape[0]
    action_bounds = [ENV.action_space.low[0], ENV.action_space.high[0]]

    agent = Agent(n_states=n_observations,
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
    
    checkpoint = torch.load("policies/RobotPolicies/FetchPickAndPlace1.pth")
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


def human_play_keyboard():
        """
        Human can play the game in real time using these keys
        """
        pressed_keys = pygame.key.get_pressed()
        action = [0.0, 0.0, 0.0, 0.0]
        move = 0.1

        if pressed_keys[pygame.K_s]: #back
             action[0] += move
        if pressed_keys[pygame.K_w]: #forward
             action[0] -= move
        if pressed_keys[pygame.K_d]: #left
             action[1] += move
        if pressed_keys[pygame.K_a]: #right
             action[1] -= move
        if pressed_keys[pygame.K_UP]: #up
             action[2] += move
        if pressed_keys[pygame.K_DOWN]: #down
             action[2] -= move
        if pressed_keys[pygame.K_c]: #gripper close
             action[3] -= move/2
        else:
             action[3] += move/2
             
        return action

import pygame

def human_play():
    """
    Human can control the robot arm's 3 joints using a joystick.
    """
    action = [0.0, 0.0, 0.0, 0.0]  #three joints + gripper control
    move = 0.25

    pygame.event.pump()
    pressed_keys = pygame.key.get_pressed()


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True 
    
    # Initialize joystick
    if pygame.joystick.get_count() > 0:
        #Get axis input to control the three joints
        joint_1 = joystick.get_axis(1)  #control X
        joint_2 = joystick.get_axis(0)  #control Y
        joint_3 = joystick.get_axis(3)  #control Z

        #Get button input for gripper control
        gripper_close = joystick.get_button(9)  #close gripper
        gripper_open = joystick.get_button(10)   #open gripper

        #Map joystick inputs to joint actions
        # print(pressed_keys)
        action[0] += joint_1 * move
        action[1] += joint_2 * move
        action[2] += -joint_3 * move

        # Gripper control
        if gripper_close:
            action[3] -= move / 4
        elif gripper_open:
            action[3] += move / 4

    return action


"""
Keeps demonstration in a python dictionary
"""
def save_demonstration(environment_name, steps, states, timestamps, rewards, seed, condition, chosen_actions, optimal_actions, timestamp_datetime):
    
    assert(len(rewards) == (len(states)) == len(timestamps) == len(optimal_actions))

    state_dict = {}

    state_dict["environment_name"] = environment_name
    state_dict["condition"] = condition
    state_dict["seed"] = seed
    state_dict["steps"] = steps
    state_dict["timestamps"] = timestamps
    state_dict["timestamps_datetime"] = timestamp_datetime
    state_dict["actions"] = chosen_actions
    state_dict["states"] = states
    state_dict["rewards"] = rewards
    state_dict["optimal_actions"] = optimal_actions
    state_dict["chosen_actions"] = chosen_actions

    return state_dict


def init_weights_biases(size):
    v = 1.0 / np.sqrt(size[0])
    return torch.FloatTensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3):
        self.n_states = n_states[0]
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=self.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = torch.tanh(self.output(x))  # TODO add scale of the action

        return output


class Critic(nn.Module):
    def __init__(self, n_states, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3, action_size=1):
        self.n_states = n_states[0]
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        self.action_size = action_size
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals + self.action_size, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=1)

    def forward(self, x, a):
        x = F.relu(self.fc1(torch.cat([x, a], dim=-1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output(x)

        return output
    
class Agent:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env,
                 k_future,
                 batch_size,
                 action_size=1,
                 tau=0.05,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.98):
        # self.device = device("cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env

        self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(device)
        self.critic = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(device)
        # self.sync_networks(self.actor)
        # self.sync_networks(self.critic)
        self.actor_target = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(device)
        self.init_target_networks()
        self.tau = tau
        self.gamma = gamma

        self.capacity = capacity
        # self.memory = Memory(self.capacity, self.k_future, self.env)

        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)

        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)

    def choose_action(self, state, goal, train_mode=True):
        state = self.state_normalizer.normalize(state)
        goal = self.goal_normalizer.normalize(goal)
        state = np.expand_dims(state, axis=0)
        goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([state, goal], axis=1)
            x = from_numpy(x).float().to(device)
            action = self.actor(x)[0].cpu().data.numpy()

        if train_mode:
            action += 0.2 * np.random.randn(self.n_actions)
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

            random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                               size=self.n_actions)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return action
    
    def init_target_networks(self):
        self.hard_update_networks(self.actor, self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)

    @staticmethod
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

        
import threading
import numpy as np
# from mpi4py import MPI


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        self.lock = threading.Lock()

    def update(self, v):
        v = v.reshape(-1, self.size)
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))
        
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / self.std, -clip_range, clip_range)


if __name__ == "__main__":
     main()

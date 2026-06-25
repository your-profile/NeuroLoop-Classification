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
import time

START_WAIT = 20000 #Time participants wait at the Enter Screen
EPISODE_WAIT = 1500

save = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

pygame.init()
pygame.time.Clock()

# Policy file is actually the policy
parser.add_argument("-c", "--condition", type=int, required=True, help="Game Condition")
parser.add_argument("-id", "--PID", type=str, required=True, help="Participant ID")
parser.add_argument("-n", "--demos", type=int, required=True, help="Number of Demonstrations")

if not os.path.exists('data/human-play-demo/'):
            os.makedirs('data/human-play-demo/')
if not os.path.exists('data/human-watch-demo/'):
            os.makedirs('data/human-watch-demo/')

args = parser.parse_args()

CONDITION_MAP = {
    0: ['w', LunarLander, "LunarLander", (4,11), "LunarLanderPolicies/LLPolicy100_1"],
    1: ['p', LunarLander, "LunarLander", (4,11), "LunarLanderPolicies/LLPolicy96"],
    2: ['w', FlappyBirdEnv, "FlappyBird", (2,12), "FlappyBirdPolicies/FlappyBirdOptimalPolicy7"],
    3: ['p', FlappyBirdEnv, "FlappyBird", (2,12), "FlappyBirdPolicies/FlappyBirdOptimalPolicy7"]
}

PARTICIPANT_ID = args.PID
CONDITION = args.condition
TASK, ENV, ENVIRONMENT_NAME, SPACE, EXPERT_FILENAME = CONDITION_MAP[args.condition]
NUM_DEMOS = args.demos

POLICY = [EXPERT_FILENAME]

n_actions, n_observations = SPACE

def main():
    if TASK == 'w':
        watch(condition=CONDITION)
    elif TASK == 'p':
        play(condition = CONDITION)
    else:
         Exception("wrong Condition Input")

def transition(x):
     rand = np.random.rand()

     if rand < 0.2:
          return np.random.randint(2, 5)
     
     return x
     
def watch(condition:int):
    #Start Time
    first_time = time.time()

    seed = np.random.randint(0,2000)
    new_seed = seed

    env = ENV(render_mode="human")

    demonstration_dict = {}

    new_seed = seed

    agent = create_agent(file=POLICY[0])

    opt_count = np.random.randint(2,5)

    for i_episode in range(0, NUM_DEMOS):
        print(opt_count)
            
        timestamps, timestamp_datetime, action_list, rewards_list, states_list, chosen_action_probs, optimal_action_probs = [], [], [], [], [], [], [] #timestamps

        state = env.reset(seed=new_seed) #reset

        if save:
            timestamps.append(time.time())
            timestamp_datetime.append(pd.to_datetime(datetime.datetime.now().timestamp() * 1000, unit='ms'))
            states_list.append(state)
            rewards_list.append(None)
            action_list.append(None)
            optimal_action_probs.append(None)
            chosen_action_probs.append(None)

        t, running_reward = 0, 0

        done = False

        if i_episode == 0:
                pygame.time.wait(START_WAIT)
                
        while not done:
            #pick an action
            agent_action, agent_action_values = agent.chooseAction(state, epsilon=0)
            optimal_action_values = agent_action_values.cpu().detach().numpy().squeeze()
            optimal_action_values = softmax(np.asarray(np.around(optimal_action_values, decimals=5)))

            # agent_action for some probabiltity want the opposite
            if (opt_count <= 0 and t > final_t):
                chosen_action_values = softmax(optimal_action_values)
                opposite_idx = np.argmin(chosen_action_values)
                optimal_idx = np.argmin(chosen_action_values)
                optimal_value = chosen_action_values[optimal_idx]
                chosen_action_values[optimal_idx] = 0.0
                chosen_action_values[opposite_idx] += (optimal_value + 0.25)
                chosen_action_values /= np.sum(np.asarray(chosen_action_values, dtype=np.float64))
                action = np.random.choice(n_actions, p=chosen_action_values)
                # print(action, agent_action)
            else:
                action = agent_action
                chosen_action_values = np.array(optimal_action_values, dtype=np.float64)

            if t > 750 and ENVIRONMENT_NAME == "LunarLander":
                action = 0

            if save:
                optimal_action_probs.append(optimal_action_values)
                chosen_action_probs.append(chosen_action_values)
            
                #timestamp update
                current_time_float = time.time()
                current_time = pd.to_datetime(datetime.datetime.now().timestamp() * 1000, unit='ms')
                timestamp_datetime.append(current_time)
                timestamps.append(current_time_float)

                action_list.append(action)

            next_state, reward, done, _ = env.step(action)
            state = next_state


            if save:

                states_list.append(state)

                rewards_list.append(reward)
                running_reward += reward

            t += 1

            if done:
                pygame.time.wait(EPISODE_WAIT)
                break

        opt_count -= 1
        
        if opt_count <= 0:
            final_t = np.random.randint(50, 100)
            opt_count = transition(opt_count)

        final_episode_actions = action_list
        final_episode_rewards = rewards_list
        final_episode_states = states_list
        final_episode_steps = t + 1

        if save:
            assert(len(timestamps) == len(final_episode_actions) == len(final_episode_rewards)== (len(final_episode_states)))
            episode = save_demonstration(environment_name=ENVIRONMENT_NAME, timestamp_datetime = timestamp_datetime, chosen_action_pd= chosen_action_probs, optimal_action_pd= optimal_action_probs, steps = final_episode_steps, states=final_episode_states, timestamps=timestamps, actions=final_episode_actions, rewards=final_episode_rewards, seed=new_seed, condition=condition)
            demonstration_dict[i_episode] = episode

        # print("Episode {} Collected Successfully".format(i_episode))
        new_seed += 1

    if save:
        demonstration_dict["NumberOfDemos"] = i_episode + 1
            
        # Convert to a hashmap
        with open('./data/human-watch-demo/{}.pickle'.format("{}_W{}_C{}".format(PARTICIPANT_ID, ENVIRONMENT_NAME[0], condition)), 'wb') as handle:
            pickle.dump(demonstration_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    last_time = time.time()

    print(last_time-first_time, " seconds long")


def play(condition:int):
    first_time = time.time()
    seed = np.random.randint(0,2000)
    new_seed = seed


    env = ENV(render_mode="human")

    agent = create_agent(POLICY[0])

    demonstration_dict = {}

    #Rewards per epsiode saved
    rewards_per_episode = []

    #for each demonstration desired
    for i_episode in range(0, NUM_DEMOS):

        timestamps, timestamp_datetime, action_list, rewards_list, states_list, chosen_action_probs, optimal_action_probs = [], [], [], [], [], [], [] #timestamps
        running_reward = 0
        
        try:
            state, _ = env.reset(seed=new_seed) #reset
        except:
            state = env.reset(seed=new_seed) #reset

        timestamps.append(time.time())
        timestamp_datetime.append(pd.to_datetime(datetime.datetime.now().timestamp() * 1000, unit='ms'))
        states_list.append(state)
        rewards_list.append(0)
        action_list.append(0)
        optimal_action_probs.append(None)
        chosen_action_probs.append(None)


        t=0
        done = False

        if i_episode == 0:
                pygame.time.wait(START_WAIT)

        while not done:
            pygame.time.wait(15)
            #human picks an action
            action = human_play()
            human_vector = np.zeros(n_actions)
            human_vector[action] = 1

            #agent chooses an action
            agent_action, agent_action_values = agent.chooseAction(state, epsilon=0)
            optimal_action_values = agent_action_values.cpu().detach().numpy().squeeze()
            optimal_action_values = softmax(np.asarray(np.around(optimal_action_values, decimals=5)))
            
            current_time_float = time.time()
            current_time = pd.to_datetime(datetime.datetime.now().timestamp() * 1000, unit='ms')

            chosen_action_probs.append(human_vector)
            action_list.append(action)
            optimal_action_probs.append(optimal_action_values)

            if ENVIRONMENT_NAME == "FlappyBird":
                next_state, reward, done, _, _ = env.step(action)
            else:
                next_state, reward, done, _ = env.step(action)

            state = next_state
            states_list.append(state)

            rewards_list.append(reward)
            running_reward += reward

            #timestamp update
            timestamps.append(current_time_float)
            timestamp_datetime.append(current_time)

            t += 1
            if done:
                rewards_per_episode.append(running_reward)
                pygame.time.wait(EPISODE_WAIT)
                break
        
        final_episode_actions = action_list
        final_episode_rewards = rewards_list
        final_episode_states = states_list
        final_episode_steps = t + 1


        assert(len(timestamps) == len(final_episode_actions) == len(final_episode_rewards)== (len(final_episode_states)))
        episode = save_demonstration(environment_name=ENVIRONMENT_NAME, steps = final_episode_steps, timestamp_datetime=timestamp_datetime, optimal_action_pd=optimal_action_probs, chosen_action_pd=chosen_action_probs, states=final_episode_states, timestamps=timestamps, actions=final_episode_actions, rewards=final_episode_rewards, seed=new_seed, condition=condition)
        demonstration_dict[i_episode] = episode

        # print("Episode {} Collected Successfully".format(i_episode))
        new_seed += 1


    demonstration_dict["NumberOfDemos"] = i_episode + 1
    # Convert to a hashmap
    with open('./data/human-play-demo/{}.pickle'.format("{}_P{}_C{}".format(PARTICIPANT_ID, ENVIRONMENT_NAME[0], condition)), 'wb') as handle:
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
 
    exp_values = np.exp(values, dtype=np.longdouble)
    exp_values_sum = np.sum(exp_values)
 
    vals = np.asarray(exp_values/exp_values_sum, dtype=np.float64)

    return vals

def KLDivergence(P, Q):
    from scipy.special import rel_entr

    return sum(rel_entr(P, Q))
    
def create_agent(file):
    agent = DQN(
        n_observations = n_observations,
        n_actions = n_actions,
        batch_size = 128,
        lr = 0.05,
        gamma = 0.99,
        mem_size = 250000,
        learn_step = 5,
        tau = 0.005,
        )
    
    checkpoint_p = torch.load('policies/{}'.format(file), map_location="cpu")
    agent.policy_net.load_state_dict(checkpoint_p['model_state_dict'])

    return agent


def human_play():
        """
        Human can play the game in real time using these keys
        """
        pygame.event.pump()
        pressed_keys = pygame.key.get_pressed()

        if ENVIRONMENT_NAME == "FlappyBird":
            if pressed_keys[pygame.K_UP]: #up
                return 1
            return 0 #do nothing
        
        else:
            # Prioritize boost so holding UP with a turn key still feels responsive.
            if pressed_keys[pygame.K_UP]: #up
                return 2
            elif pressed_keys[pygame.K_LEFT]: #left
                return 1
            elif pressed_keys[pygame.K_RIGHT]: #right
                return 3
            return 0 #do nothing

"""
Keeps demonstration in a python dictionary
"""
def save_demonstration(environment_name, steps, states, timestamps, actions, rewards, seed, condition, chosen_action_pd, optimal_action_pd, timestamp_datetime):
    
    assert(len(rewards) == (len(states)) == len(timestamps) == len(actions))

    state_dict = {}

    state_dict["environment_name"] = environment_name
    state_dict["condition"] = condition
    state_dict["seed"] = seed
    state_dict["steps"] = steps
    state_dict["timestamps"] = timestamps
    state_dict["timestamps_datetime"] = timestamp_datetime
    state_dict["states"] = states
    state_dict["actions"] = actions
    state_dict["rewards"] = rewards
    state_dict["chosen_actions"] = chosen_action_pd
    state_dict["optimal_actions"] = optimal_action_pd

    return state_dict

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

if __name__ == "__main__":
     main()

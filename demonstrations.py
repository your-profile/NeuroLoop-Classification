import time
import pygame
import pickle

"""
NAME
Started: 06/01/23
Last Updated: 12/21/23

Functions that save and replay demonstrations from trained and human agents.
"""
class Demonstrations():
    def __init__(self, env_name:None, filename=None, agent=None, env=None, number_of_demonstrations=None, seed=None, experiment=False):

        self.demo_file_name = filename
        self.agent = agent
        self.env = env
        
        self.demonstration_dict = {"demo_name": self.demo_file_name, "algorithm": "DeepQLearning"} #dictionary of demonstrations
        self.num_demos = number_of_demonstrations
        self.envNAME = env_name
        self.seed = seed
        self.experiment = experiment
        pygame.init()


    def human_play(self):
        """
        Human can play the game in real time using these keys
        """
        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[pygame.K_SPACE]:
            input("Press Any Key to Resume the Game!")

        if self.envNAME == "lunar_lander":
            if pressed_keys[pygame.K_LEFT]: #left
                return 1
            elif pressed_keys[pygame.K_UP]: #up
                return 2
            elif pressed_keys[pygame.K_RIGHT]: #right
                return 3
            return 0 #do nothing
        elif self.envNAME == "snake":
            if pressed_keys[pygame.K_LEFT]: #left
                return 0
            elif pressed_keys[pygame.K_UP]: #up
                return 2
            elif pressed_keys[pygame.K_RIGHT]: #right
                return 1
            elif pressed_keys[pygame.K_DOWN]:
                return 3
            else:
                return None
        elif self.envNAME == "car racing":
            if pressed_keys[pygame.K_RIGHT]:
                return 1
            if pressed_keys[pygame.K_LEFT]:
                return 2
            if pressed_keys[pygame.K_UP]:
                return 3
            if pressed_keys[pygame.K_DOWN]:
                return 4
            else: return 0

        if pressed_keys[pygame.K_SPACE]:
            input("Press Any Key to Resume the Game!")
        


    """
    Collects demonstrations from human agents
    """
    def collect_demonstrations(self):
        #Rewards per epsiode saved
        rewards_per_episode = []
        seed_change = self.seed
        steps = 600 #self.env.steps
        
        try:
            self.env = self.env(render_mode = "human")
        except:
            self.env = self.env()

        #for each demonstration desired
        for i_episode in range(0, self.num_demos):
            seed_change += i_episode
            print("seed:", seed_change)

            timestamps, action_list, rewards_list, states_list = [], [], [], [] #timestamps

            running_reward = 0

            try: 
                state = self.env.reset(seed=seed_change) #reset
            except:
                state = self.env.reset() #reset

            self.env.render()

            timestamps.append(time.time())
            states_list.append(state)
            rewards_list.append(0)
            action_list.append(0)
            
            human_action = 0
            t=0

            print(i_episode)

            done = False

            while not done:
                #pick an action
                action = self.human_play()

                if action is not None:
                    human_action = action
                
                action_list.append(human_action)

                states_list.append(state)

                next_state, reward, done, win = self.env.step(human_action)

                rewards_list.append(reward)

                running_reward += reward
                state = next_state

                #timestamp update
                current_time = time.time()
                timestamps.append(current_time)

                t += 1
                if done:
                    rewards_per_episode.append(running_reward)
                    states_list.append(state)
                    break
            
            final_episode_actions = action_list
            final_episode_rewards = rewards_list
            final_episode_states = states_list
            final_episode_steps = t + 1


            assert(len(timestamps) == len(final_episode_actions) == len(final_episode_rewards)== (len(final_episode_states)-1))
            episode = self.save_demonstration(environment_name="lunar lander", steps = final_episode_steps, states=final_episode_states, timestamps=timestamps, actions=final_episode_actions, rewards=final_episode_rewards, seed=seed_change, environment_version=1.0)
            self.demonstration_dict[i_episode] = episode

            print("Episode {} Collected Successfully".format(i_episode))

        self.demonstration_dict["NumberOfDemos"] = i_episode + 1
        self.demonstration_dict["StepLimit"] = steps

    """
    Collects demonstrations from trained agents
    """
    def collect_agent_demonstrations(self, number, agent):
        env = self.env()
        #Rewards per epsiode saved
        rewards_per_episode = []
        seed_change = self.seed
        steps = 600 #self.env.steps

        #for each demonstration desired
        for i_episode in range(0, number):
            print(i_episode)
            seed_change += i_episode

            timestamps, action_list, rewards_list, states_list = [], [], [], [] #timestamps

            running_reward = 0

            state = env.reset(seed=seed_change) #reset

            t, wins = 0,0


            done = False

            while not done:
                action = agent.chooseAction(state, epsilon=0)
                action_list.append(action)

                states_list.append(state)

                next_state, reward, done, win = env.step(action)

                rewards_list.append(reward)

                running_reward += reward
                state = next_state

                #timestamp update
                current_time = time.time()
                timestamps.append(current_time)

                t += 1

                if done:
                    rewards_per_episode.append(running_reward)
                    states_list.append(state)
                    if win:
                        wins +=1
                    break
            
            final_episode_actions = action_list
            final_episode_rewards = rewards_list
            final_episode_states = states_list
            final_episode_steps = t + 1


            assert(len(timestamps) == len(final_episode_actions) == len(final_episode_rewards)== (len(final_episode_states)-1))
            episode = self.save_demonstration(environment_name="lunar lander", steps = final_episode_steps, states=final_episode_states, timestamps=timestamps, actions=final_episode_actions, rewards=final_episode_rewards, seed=seed_change, environment_version=1.0)
            self.demonstration_dict[i_episode] = episode


        self.demonstration_dict["NumberOfDemos"] = i_episode + 1
        self.demonstration_dict["StepLimit"] = steps

        return wins/number


    """
    Saves demonstration in a python dictionary
    """
    def save_demonstration(self, environment_name, steps, states, timestamps, actions, rewards, seed, environment_version):
        
        assert(len(rewards) == (len(states) - 1) == len(timestamps) == len(actions))

        state_dict = {}

        state_dict["environment_name"] = environment_name
        state_dict["environment_version"] = environment_version
        state_dict["seed"] = seed
        state_dict["steps"] = steps
        state_dict["timestamps"] = timestamps
        state_dict["states"] = states
        state_dict["actions"] = actions
        state_dict["rewards"] = rewards

        return state_dict
    

    """
    Replays a saved demonstration
    """

    def play_demonstrations(self, demo_dict):
        env = self.env(render_mode = "human")
        wins = 0

        print("NUMDEMOS",demo_dict["NumberOfDemos"])
        for i in range(demo_dict["NumberOfDemos"]):
            steps = demo_dict[i]["steps"]
            seed = demo_dict[i]["seed"]
            state = env.reset(seed=seed)

            for j in range(steps):
                action = demo_dict[i]["actions"][j]
                
                #Return state, reward
                _, _, done, win = env.step(action)

                if done:
                    if win:
                        print("won", i)
                        wins+=1
                    break
        print(wins/demo_dict["NumberOfDemos"])

    def evaluate_demonstrations(self, demo_dict):
            env = self.env()
            wins = 0

            for i in range(demo_dict["NumberOfDemos"]):
                steps = demo_dict[i]["steps"]
                seed = demo_dict[i]["seed"]
                state = env.reset(seed=seed)

                for j in range(steps):
                    action = demo_dict[i]["actions"][j]
                    
                    #Return state, reward
                    _, _, done, win = env.step(action)

                    if done:
                        if win:
                            print("won", i)
                            wins+=1
                        break
            print("Success Rate: ", wins/demo_dict["NumberOfDemos"])

    """
    Save to pickle file in Data/Demonstration folder
    """
    def save_demo(self):
        with open('./data/demonstrations/{}.pickle'.format(self.demo_file_name), 'wb') as handle:
            pickle.dump(self.demonstration_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved as:", self.demo_file_name)
        

    def collectDemos(self):
        print("Collecting Demonstrations!")

        self.collect_demonstrations()
        self.save_demo()

        print("Demonstrations Collected!")

    def collectAgentDemos(self, num_demos, agent):
        print("Collecting Demonstrations!")

        self.collect_agent_demonstrations(num_demos, agent)
        self.save_demo()

        dict = self.readDemos()

        self.play_demonstrations(dict)

        print("Demonstrations Collected!")

    def readDemos(self, filename=None):

        file = open('./data/demonstrations/{}.pickle'.format(self.demo_file_name), 'rb')
        demo_dict = pickle.load(file)
        file.close()

        return demo_dict

    def renderDemos(self, demo_file_name=None):
        if demo_file_name is None:
            dict = self.readDemos(self.demo_file_name)
        else: 
            dict = self.readDemos(demo_file_name)

        self.play_demonstrations(dict)
        self.evaluate_demonstrations(dict)

    def evalDemos(self, demo_file_name=None):
        if demo_file_name is None:
            dict = self.readDemos(self.demo_file_name)
        else: 
            dict = self.readDemos(demo_file_name)

        self.evaluate_demonstrations(dict)





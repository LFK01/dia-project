import numpy as np


class Learner:
    def __init__(self, n_environments, n_arms):
        # store the number of environments
        self.n_environments = n_environments
        # store the number of arms used by the learner
        self.n_arms = n_arms
        # initialize the turn variable
        self.t = 0
        # initialize the rewards structure used to store the rewards of each arm for each environment
        self.rewards_per_arm = x = [[[] for i in range(n_arms)] for i in range(n_environments)]
        # initialize the rewards array used to store the general rewards
        self.collected_rewards = np.array([])

    def pull_arm(self):
        pass

    def update_observations(self, pulled_arm, reward):
        # list used to iterate over the environments
        environment_indexes_list = range(len(pulled_arm))
        # iterate over the environments
        for env in environment_indexes_list:
            # store the reward of the corresponding arm in the appropriate list
            self.rewards_per_arm[pulled_arm[env][0]][pulled_arm[env][1]].append(reward[env])
        # update the numpy array
        self.collected_rewards = np.append(self.collected_rewards, reward)

import numpy as np


# Learner class
class Learner:

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    # Update the observations made by the learner. Given 'pulled_arm' and 'reward':
    # - append the value of 'reward' to the rewards collected by 'pulled_arm'
    # - append the value of 'reward' to the general 'collected_rewards' numpy array
    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    # Do nothing
    def pull_arm(self):
        pass

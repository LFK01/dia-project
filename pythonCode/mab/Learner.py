import numpy as np


# class Learner:
#     def __init__(self, n_environments, n_arms):
#         # store the number of environments
#         self.n_environments = n_environments
#         # store the number of arms used by the learner
#         self.n_arms = n_arms
#         # initialize the turn variable
#         self.t = 0
#         # initialize the rewards structure used to store the rewards of each arm for each environment
#         self.rewards_per_arm = x = [[[] for i in range(n_arms)] for i in range(n_environments)]
#         # initialize the rewards array used to store the general rewards
#         self.collected_rewards = [[] for i in range(n_environments)]

class Learner:

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def pull_arm(self):
        pass

    # def update_observations(self, pulled_arm, reward):
    #     # iterate over the environments
    #     for env in range(self.n_environments):
    #         # store the reward of the corresponding arm in the appropriate list
    #         self.rewards_per_arm[pulled_arm[env][0]][pulled_arm[env][1]].append(reward[env])
    #         # update the rewards array
    #         self.collected_rewards[env].append(reward[env])

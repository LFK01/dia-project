import numpy as np


# Environment class used to build the Multi-Armed-Bandit environment
class Environment:
    # constructor method
    def __init__(self, n_arms, probabilities):
        # n_arms contains the integer number of arms
        self.n_arms = n_arms
        # probabilities is a vector of probabilities which defines how likely is an arm to give a reward
        # Example: [array([0.        , 0.03448276, ... 0.96551724, 1.        ]),
        #           array([0.        , 0.02413793, ... 0.67586207, 0.7       ]),
        #           array([0.        , 0.01034483, ... 0.28965517, 0.3       ])]
        self.probabilities = probabilities

    # function round returns the reward given from pulled_arm
    def round(self, pulled_arms):
        # list used to store the rewards given by the selected arms in the superarm
        rewards = []
        # list used to iterate over the environments
        environment_indexes_list = range(0, len(self.probabilities))
        # iterate over the environments
        for env in environment_indexes_list:
            # retrieve the reward corresponding to the selected arm
            rewards.append(np.random.binomial(1, self.probabilities[pulled_arms[env][0]][pulled_arms[env][1]]))
        # return the list of obtained rewards
        return rewards

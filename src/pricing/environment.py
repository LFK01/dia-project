import numpy as np


# environment class used to build the Multi-Armed-Bandit environment
class Environment:
    # constructor method
    def __init__(self, n_arms, probabilities):
        # n_arms contains the integer number of arms
        self.n_arms = n_arms
        # probabilities is a vector of probabilities which defines how likely is an arm to give a reward
        # Example: [array([0.        , 0.03448276, ... 0.96551724, 1.        ])]
        self.probabilities = probabilities

    # function round returns the reward given from pulled_arm
    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

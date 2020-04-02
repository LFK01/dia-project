import numpy as np


# Environment class used to build the Multi-Armed-Bandit environment
class Environment:
    # constructor method
    def __init__(self, n_arms, probabilities, budgets):
        # n_arms contains the integer number of arms
        self.n_arms = n_arms
        # probabilities is a vector of probabilities which defines how likely is an arm to give a reward
        self.probabilities = probabilities
        self.budgets = budgets

    # function round returns the reward given from pulled_arm
    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])*self.budgets[pulled_arm]
        return reward

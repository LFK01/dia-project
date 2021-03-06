import numpy as np
from src.utils.learner import Learner


class TSLearner(Learner):
    def __init__(self, n_arms, probabilities, number_of_classes, prices=None):
        super().__init__(n_arms)
        # initialize the parameters of the beta distribution
        # Example: [[[1. 1.], ... [1. 1.], [1. 1.]]]
        self.beta_parameters = np.ones((number_of_classes, n_arms, 2))
        # store the matrix of prices for each environment
        # Example: [array([0. , 0.03448276, ... 0.96551724, 1.]),
        #           array([0. , 0.03448276, ... 0.96551724, 1.]),
        #           array([0. , 0.03448276, ... 0.96551724, 1.])]
        self.rewards_per_arm = [[[] for i in range(n_arms)] for j in range(number_of_classes)]
        self.collected_rewards = [[] for i in range(number_of_classes)]
        self.prices = prices
        self.probabilities = probabilities
        self.number_of_classes = number_of_classes

    def pull_arm(self):
        scores = np.zeros((self.number_of_classes, self.n_arms))
        for cls in range(0, self.number_of_classes):
            scores[cls] = (np.random.beta(self.beta_parameters[cls, :, 0], self.beta_parameters[cls, :, 1]) *
                           self.probabilities[cls]) * self.prices
        idx = np.argmax(scores.sum(axis=0))
        return idx

    def update(self, pulled_arm, rewards):
        self.t += 1
        self.update_observations(pulled_arm, rewards)
        for cls in range(0, self.number_of_classes):
            reward = 0
            if rewards[cls] > 0:
                reward = 1
            self.beta_parameters[cls][pulled_arm, 0] = self.beta_parameters[cls][pulled_arm, 0] + reward
            self.beta_parameters[cls][pulled_arm, 1] = self.beta_parameters[cls][pulled_arm, 1] + 1.0 - reward

    def update_observations(self, pulled_arm, reward):
        for cls in range(0, self.number_of_classes):
            self.rewards_per_arm[cls][pulled_arm].append(reward[cls])
            self.collected_rewards[cls].append(reward[cls])

    def get_conversion_rate(self, index):
        scores = np.zeros((self.number_of_classes, self.n_arms))
        for cls in range(0, self.number_of_classes):
            scores[cls] = (np.random.beta(self.beta_parameters[cls, :, 0], self.beta_parameters[cls, :, 1]) *
                           self.probabilities[cls])
        conversion_rate_vector = []
        for cls in range(0, self.number_of_classes):
            conversion_rate_vector.append(scores[cls, index])
        return conversion_rate_vector

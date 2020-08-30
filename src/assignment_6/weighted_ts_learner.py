import numpy as np
from src.utils.learner import Learner


class WeightedTSLearner(Learner):
    def __init__(self, n_arms, prices):
        super().__init__(n_arms)
        # initialize the parameters of the beta distribution
        # Example: [[[1. 1.], ... [1. 1.], [1. 1.]]]
        self.beta_parameters = np.ones((n_arms, 2))
        # store the array of prices for each environment
        # Example: [0., 0.03448276, ... 0.96551724, 1.]
        self.prices = prices

    def pull_arm(self):
        conversion_rates = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        weighting_factor = conversion_rates * self.prices
        idx = np.argmax(weighting_factor)
        return idx, np.max(conversion_rates)

    def update(self, pulled_arm, reward_in_money):
        self.t += 1
        self.update_observations(pulled_arm, reward_in_money)
        reward_beta_dist = 0
        if reward_in_money > 0:
            reward_beta_dist = 1
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward_beta_dist
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward_beta_dist

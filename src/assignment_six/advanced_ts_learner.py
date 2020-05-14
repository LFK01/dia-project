from src.advertising.learner.learner import Learner
import numpy as np


class AdvancedTSLearner(Learner):
    def __init__(self, n_arms, prices):
        super().__init__(n_arms)
        # initialize the parameters of the beta distribution
        # Example: [[[1. 1.],  ...  [1. 1.],  [1. 1.]]]
        self.beta_parameters = np.ones((n_arms, 2))
        # store the matrix of prices for each environment
        # Example: [array([0.        , 0.03448276, ... 0.96551724, 1.        ]),
        #           array([0.        , 0.03448276, ... 0.96551724, 1.        ]),
        #           array([0.        , 0.03448276, ... 0.96551724, 1.        ])]
        self.prices = prices

    def pull_arm(self):
        conversion_rates = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        idx = np.argmax(conversion_rates)
        return idx, np.max(conversion_rates)

    def get_price_from_index(self, idx):
        return self.prices[idx]

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward


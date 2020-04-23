import numpy as np
from src.advertising.learner.learner import Learner


class TSLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # initialize the parameters of the beta distribution
        # Example: [[[1. 1.],  ...  [1. 1.],  [1. 1.]]]
        self.beta_parameters = np.ones((n_arms, 2))
        # store the matrix of prices for each environment
        # Example: [array([0.        , 0.03448276, ... 0.96551724, 1.        ]),
        #           array([0.        , 0.03448276, ... 0.96551724, 1.        ]),
        #           array([0.        , 0.03448276, ... 0.96551724, 1.        ])]

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

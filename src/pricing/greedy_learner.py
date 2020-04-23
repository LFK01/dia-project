import numpy as np
from src.advertising.learner.learner import Learner


class GreedyLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # expected payoff is a matrix containing the  expected payoff of each arm
        self.expected_payoffs = np.zeros(n_arms)

    def update(self, pulled_arm, reward):
        # update round number
        self.t += 1
        # update the observations of the rewards
        self.update_observations(pulled_arm, reward)
        self.expected_payoffs[pulled_arm] = (self.expected_payoffs[pulled_arm] * (self.t - 1.0) + reward) / self.t

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        indexes = np.argwhere(self.expected_payoffs == np.amax(self.expected_payoffs)).reshape(-1)
        pulled_arm = np.random.choice(indexes)
        return pulled_arm

import numpy as np
from src.utils.learner import Learner


class GreedyLearner(Learner):
    def __init__(self, n_arms, probabilities, number_of_classes, prices=None):
        super().__init__(n_arms)
        # expected payoff is a matrix containing the  expected payoff of each arm
        self.rewards_per_arm = [[[] for i in range(n_arms)] for j in range(number_of_classes)]
        self.collected_rewards = [[] for i in range(number_of_classes)]
        self.expected_payoffs = np.zeros((number_of_classes, n_arms))
        self.prices = prices
        self.probabilities = probabilities
        self.number_of_classes = number_of_classes

    def update(self, pulled_arm, rewards_vector):
        # update round number
        self.t += 1
        # update the observations of the rewards
        self.update_observations(pulled_arm, rewards_vector)
        for cls in range(self.number_of_classes):
            reward = 0
            if rewards_vector[cls] > 0:
                reward = 1
            self.expected_payoffs[cls, pulled_arm] = (self.expected_payoffs[cls, pulled_arm]
                                                      * (self.t - 1.0) + reward) / self.t

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        indexes = np.argwhere(np.average(self.expected_payoffs, axis=0, weights=self.probabilities) * self.prices ==
                              np.amax(np.average(self.expected_payoffs, axis=0,
                                                 weights=self.probabilities) * self.prices)).reshape(-1)
        pulled_arm = np.random.choice(indexes)
        return pulled_arm

    def update_observations(self, pulled_arm, reward):
        for cls in range(0, self.number_of_classes):
            self.rewards_per_arm[cls][pulled_arm].append(reward[cls])
            self.collected_rewards[cls].append(reward[cls])

from src.pricing.ts_learner import TSLearner
import numpy as np


class AdvancedTSLearner(TSLearner):
    def __init__(self, n_arms, prices):
        super().__init__(n_arms)
        self.prices = prices

    def pull_arm(self):
        conversion_rates = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        idx = np.argmax(conversion_rates)
        return idx, np.max(conversion_rates)

    def get_price_from_index(self, idx):
        return self.prices[idx]
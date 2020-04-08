import numpy as np


# Conversion rate curve:
# given the budget spent, returns number of clicks
def clicks(budget):
    return 100 * (1.0 - np.exp(-4 * budget + 3 * budget ** 3))


# Generate a random sample based on the curve:
# given a budget and a noise, returns number of clicks + random gaussian noise
def generate_observation(budget, noise_std):
    return clicks(budget) + np.random.normal(0, noise_std, size=clicks(budget).shape)


# ClickBudget class
class ClickBudget:
    def __init__(self, budget_id, budgets, sigma):
        self.budgets = budgets
        self.means = clicks(budgets)
        self.sigmas = np.ones(len(budgets)) * sigma
        self.budget_id = budget_id

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

    def get_clicks_function(self):
        return self

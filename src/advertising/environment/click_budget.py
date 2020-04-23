import numpy as np


# Conversion rate curve:
# given the budget spent, returns number of clicks
# a = 100, b = 1.0, c = 4, d = 3, e = 3
def clicks(budget, a, b, c, d, e):
    return a * (b - np.exp(-c * budget + d * budget ** e))


# Generate a random sample based on the curve:
# given a budget and a noise, returns number of clicks + random gaussian noise
def generate_observation(budget, noise_std):
    return clicks(budget) + np.random.normal(0, noise_std, size=clicks(budget).shape)


# ClickBudget class
class ClickBudget:
    def __init__(self, budget_id, budgets, sigma, a=100, b=1.0, c=4, d=3, e=3):
        self.budgets = budgets
        self.means = clicks(budgets, a, b, c, d, e)
        self.sigmas = np.ones(len(budgets)) * sigma
        self.budget_id = budget_id

    def round(self, pulled_arm):
        # Returning the rewards avoiding negative value
        return np.maximum(0, np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm]))

    def get_clicks_function(self):
        return self

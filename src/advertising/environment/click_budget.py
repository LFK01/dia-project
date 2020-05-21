import numpy as np


# Conversion rate curve:
# given the budget spent, returns number of clicks
# a = 100, b = 1.0, c = 4, d = 3, e = 3
def clicks(budget, max_value, coefficient, function_type):
    if function_type == 0:
        return max_value * (1.0 - np.exp(-coefficient * budget + 7 / 2 * budget ** 2))
    if function_type == 1:
        return max_value * (1.0 - np.exp(-coefficient * budget + 3 * budget ** 3))
    if function_type == 2:
        return max_value * (1.0 - np.exp(-coefficient * budget
                                         + 7 / 2 * budget ** 2 + 3 * budget ** 3 - 3 * budget ** 4))
    return 1


# Generate a random sample based on the curve:
# given a budget and a noise, returns number of clicks + random gaussian noise
def generate_observation(budget, arm, max_value=100, coefficient=4, function_type=0, noise_std=10.0):
    means = clicks(budget, arm, max_value, coefficient, function_type)
    sigmas = np.ones(len(budget)) * noise_std
    return np.maximum(0, np.random.normal(means, sigmas))


# Modify the curve
def modify_curve(means, function_type):
    if function_type == 0:
        return means
    if function_type == 1:
        for m in range(0, len(means)):
            if 3 <= m <= 5:
                means[m] /= 2
        return means
    if function_type == 2:
        for m in range(0, len(means)):
            if 0 <= m <= 4:
                means[m] /= 3
        return means


# ClickBudget class
class ClickBudget:
    def __init__(self, budget_id, budgets, sigma, max_value=100, coefficient=4, function_type=0):
        self.budgets = budgets
        self.means = modify_curve(clicks(budgets, max_value, coefficient, function_type), function_type)
        self.sigmas = np.ones(len(budgets)) * sigma
        self.budget_id = budget_id

    def round(self, pulled_arm):
        # Returning the rewards avoiding negative value
        return np.maximum(0, np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm]))

    def get_clicks_function(self):
        return self

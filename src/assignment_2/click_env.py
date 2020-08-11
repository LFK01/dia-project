import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


# Conversion rate curve:
# given the budget spent, returns number of clicks
# a = 100, b = 1.0, c = 4, d = 3, e = 3
def clicks(budgets, function):
    return function(budgets)


# Generate a random sample based on the curve:
# given a budget and a noise, returns number of clicks + random gaussian noise
def generate_observation(budget, function, noise_std=10.0):
    means = clicks(budget, function)
    sigmas = np.ones(len(budget)) * noise_std
    return np.maximum(0, np.random.normal(means, sigmas))


# ClickEnv class
class ClickEnv:
    def __init__(self, budgets, sigma, x_values, y_values, subcampaign_number, color='g'):
        self.budgets = budgets
        self.function = interpolate.interp1d(x_values, y_values)
        plt.ylabel("Rewards")
        plt.xlabel("arms")
        plt.plot(x_values, y_values, color)
        plt.legend(["environment function of the subcampaign " + str(subcampaign_number)])
        plt.show()
        self.means = clicks(budgets, self.function)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        # Returning the rewards avoiding negative value
        return np.maximum(0, np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm]))

    def get_clicks_function(self):
        return self

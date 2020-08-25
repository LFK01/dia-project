import os
from src.utils.constants import subcampaign_names, img_path
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


# Conversion rate curve:
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
        plt.ylabel("Expected number of clicks")
        plt.xlabel("Percentage of daily budget allocated")
        x_max = np.max(x_values)
        x_min = np.min(x_values)
        x = np.linspace(x_max, x_min, 100)
        y = [self.function(x[i]) for i in range(0, 100)]
        plt.plot(x, y, color)
        plt.legend(["Subcampaign " + str(subcampaign_number) + " " + subcampaign_names[subcampaign_number - 1]])
        img_name = "subcampaign_" + str(subcampaign_number) + ".png"
        plt.savefig(os.path.join(img_path, img_name))
        plt.show()
        self.means = clicks(budgets, self.function)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        # Returning the rewards avoiding negative value
        return np.maximum(0, np.random.normal(0, self.sigmas[pulled_arm]) + self.means[pulled_arm])

    def get_clicks_function(self):
        return self

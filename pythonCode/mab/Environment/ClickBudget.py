import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class ClickBudget:
    def __init__(self, id, budgets, sigma):
        self.budgets = budgets
        self.means = self.clicks(budgets)
        self.sigmas = np.ones(len(budgets)) * sigma
        self.id = id

    # Return the estimated n of
    def clicks(self, budget):
        return 100 * (1.0 - np.exp(-4 * budget + 3 * budget ** 3))

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

    def getClicksFunction(self):
        return self

    def generate_observations(self, x, noise_std):
        return self.clicks(x) + np.random.normal(0, noise_std, size=self.clicks(x).shape)

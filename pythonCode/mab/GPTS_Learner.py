import numpy as np
from sklearn import preprocessing
from pythonCode.mab.Learner import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms):
        super(GPTS_Learner, self).__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        self.alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha ** 2, normalize_y=True,
                                           n_restarts_optimizer=9)
        self.x_obs = np.array([])
        self.y_obs = np.array([])

    def update_observations(self, arm_idx, reward):
        super(GPTS_Learner, self).update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)

    def get_predicted_arm(self, idx):
        if idx == 0:
            return 0
        predicted_arms = self.means[idx]
        return predicted_arms

    def generate_gaussian_process(self, new_x_obs, new_y_obs):
        self.x_obs = np.append(self.x_obs, new_x_obs)
        self.y_obs = np.append(self.y_obs, new_y_obs)

        X = np.atleast_2d(self.x_obs).T
        Y = self.y_obs.ravel()

        kernel = self.gp.kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha ** 2, normalize_y=True,
                                           n_restarts_optimizer=9)
        self.gp.fit(X, Y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.means = np.maximum(0, self.means)

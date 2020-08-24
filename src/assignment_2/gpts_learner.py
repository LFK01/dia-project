import numpy as np
from src.utils.learner import Learner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


# GP-TS Learner, subclass of Learner
class GPTSLearner(Learner):
    def __init__(self, n_arms, arms):
        super(GPTSLearner, self).__init__(n_arms)
        self.time = 0
        self.arms = arms
        self.predicted_arms = np.zeros(self.n_arms)
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        self.alpha = 2
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha=self.alpha ** 2,
                                           normalize_y=True,
                                           n_restarts_optimizer=9)
        self.x_obs = np.array([])
        self.y_obs = np.array([])

    # Update the learner observation with the last arm and reward chosen
    def update_observations(self, arm_idx, reward):
        super(GPTSLearner, self).update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    # Update the Gaussian Process model with the last observed arm and reward
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards

        # Normalization of X
        # x = preprocessing.scale(x)

        # Fit the model
        self.update_prediction(x, y)

    # Run both the update function, increasing the round
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    # Pull all the sample from the learner
    def pull_arm(self):
        sampled_values = self.predicted_arms
        # If the sum of the predicted_arms is 0, initialize with small value
        if sum(sampled_values) == 0:
            sampled_values = [i * 1e-3 for i in range(len(self.arms))]
        # Set the predicted rewards for budget = 0 to 0
        sampled_values[0] = 0
        return sampled_values

    # Update the prediction accordingly to the new input data
    def update_prediction(self, x, y):
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        # self.sigmas = np.maximum(self.sigmas, 1e-2)
        self.sigmas = np.maximum(self.sigmas, 1)

        # Save the predicted value for each arms, avoiding negative value
        self.predicted_arms = np.random.normal(self.means, self.sigmas)
        self.predicted_arms = np.maximum(0, self.predicted_arms)

    # Generate a Gaussian Process with the observed value (not used but will be useful in future)
    def generate_gaussian_process(self, new_x_obs, new_y_obs, reset_gp=False):
        self.x_obs = np.array([])
        self.y_obs = np.array([])
        self.x_obs = np.append(self.x_obs, new_x_obs)
        self.y_obs = np.append(self.y_obs, new_y_obs)

        x = np.atleast_2d(self.x_obs).T
        y = self.y_obs

        if reset_gp:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
            self.gp = GaussianProcessRegressor(kernel=kernel,
                                               alpha=self.alpha ** 2,
                                               normalize_y=True,
                                               n_restarts_optimizer=9)
        self.gp.fit(x, y)

        kernel = self.gp.kernel_
        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha=self.gp.alpha,
                                           normalize_y=True,
                                           n_restarts_optimizer=0)

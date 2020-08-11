import numpy as np
from src.assignment_2.gpts_learner import GPTSLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


# GP-TS Learner, subclass of Learner
class GPTSLearnerSW(GPTSLearner):
    def __init__(self, n_arms, arms, window_size):
        super(GPTSLearnerSW, self).__init__(n_arms, arms)
        self.window_size = window_size

    # Update the learner observation with the last arm and reward chosen
    def update_observations(self, arm_idx, reward):
        super(GPTSLearnerSW, self).update_observations(arm_idx, reward)

    # Update model in sliding windows mode
    def update_model(self):
        # Get only the pulled_arms and collected rewards inside the sliding window
        pulled_arms_in_window = self.pulled_arms[-self.window_size:]
        collected_reward_in_window = self.collected_rewards[-self.window_size:]

        # Create variable x and y accordingly to what is needed to gp.fit()
        x = np.atleast_2d(pulled_arms_in_window).T
        y = collected_reward_in_window

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

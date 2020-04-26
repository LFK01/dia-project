from src.advertising.learner.learner import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import preprocessing


# GP-TS Learner, subclass of Learner
class GPTSLearner(Learner):
    def __init__(self, n_arms, arms):
        super(GPTSLearner, self).__init__(n_arms)
        self.arms = arms
        self.predicted_arms = np.zeros(self.n_arms)
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        self.alpha = 10.0
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
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

        # Save the predicted value for each arms, avoiding negative value
        self.predicted_arms = np.random.normal(self.means, self.sigmas)
        self.predicted_arms = np.maximum(0, self.predicted_arms)

    # Run both the update function, increasing the round
    def update(self, pulled_arm, reward, window_size=0, sw=False):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        if sw:
            self.update_model_sw(window_size)
        else:
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

    # update model in sliding windows mode
    def update_model_sw(self, window_size):
        # for arm in range(0, len(self.arms)):
        #     n_sample = np.sum(self.pulled_arms[-window_size:] == arm)
        #     collected_reward_in_window = self.collected_rewards[arm][-n_sample:]

        # Get only the pulled_arms and collected rewards inside the sliding window
        pulled_arms_in_window = self.pulled_arms[-window_size:]
        collected_reward_in_window = self.collected_rewards[-window_size:]
        x = np.atleast_2d(pulled_arms_in_window).T
        y = collected_reward_in_window

        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

        # Save the predicted value for each arms, avoiding negative value
        self.predicted_arms = np.random.normal(self.means, self.sigmas)
        self.predicted_arms = np.maximum(0, self.predicted_arms)

    # Generate a Gaussian Process with the observed value (not used but will be useful in future)
    def generate_gaussian_process(self, new_x_obs, new_y_obs):
        self.x_obs = np.append(self.x_obs, new_x_obs)
        self.y_obs = np.append(self.y_obs, new_y_obs)

        x = np.atleast_2d(self.x_obs).T
        y = self.y_obs.ravel()
        kernel = self.gp.kernel
        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha=self.alpha ** 2,
                                           normalize_y=True,
                                           n_restarts_optimizer=10)
        self.gp.fit(x, y)

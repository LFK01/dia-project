from pythonCode.mab.Learner import *


class Greedy_Learner(Learner):
    def __init__(self, n_environments, n_arms):
        super().__init__(n_environments, n_arms)
        self.expected_payoffs = np.zeros((n_environments, n_arms))

    def update(self, pulled_arm, reward):
        # update round number
        self.t += 1
        # update the observations of the rewards
        self.update_observations(pulled_arm, reward)
        # iterate over the environments
        for env in range(self.n_environments):
            # update the expected payoffs
            self.expected_payoffs[pulled_arm[env][0]][pulled_arm[env][1]] = \
                (self.expected_payoffs[pulled_arm[env][0]][pulled_arm[env][1]] * (self.t - 1.0) + reward[env]) / self.t

    def pull_arm(self):
        if self.t < self.n_arms:
            # list used to store the arms selected in the initial turns
            initial_turns_arms = []
            # iterate over the environments
            for environment in range(self.n_environments):
                # build the list of arms
                initial_turns_arms.append([environment, self.t])
            return initial_turns_arms
        # retrieve the indexes of the most promising arms
        pulled_arms = []
        # iterate over the environments
        for environment in range(self.n_environments):
            # retrieve the best arms
            indexes = np.argwhere(self.expected_payoffs[environment] == self.expected_payoffs[environment].max())\
                .reshape(-1)
            # break ties and store the coordinate
            pulled_arms.append([environment, np.random.choice(indexes)])
            # return the superarm
        return pulled_arms

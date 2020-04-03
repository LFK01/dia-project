class Learner:
    def __init__(self, n_environments, n_arms):
        # store the number of environments
        self.n_environments = n_environments
        # store the number of arms used by the learner
        self.n_arms = n_arms
        # initialize the turn variable
        self.t = 0
        # initialize the rewards structure used to store the rewards of each arm for each environment
        self.rewards_per_arm = x = [[[] for i in range(n_arms)] for i in range(n_environments)]
        # initialize the rewards array used to store the general rewards
        self.collected_rewards = [[] for i in range(n_environments)]

    def pull_arm(self):
        pass

    def update_observations(self, pulled_arm, reward):
        # iterate over the environments
        for env in range(self.n_environments):
            # store the reward of the corresponding arm in the appropriate list
            self.rewards_per_arm[pulled_arm[env][0]][pulled_arm[env][1]].append(reward[env])
            # update the rewards array
            self.collected_rewards[env].append(reward[env])

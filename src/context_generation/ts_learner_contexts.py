from src.advertising.learner.learner import *


# This class allows to do the usual things of a thompson sampling but in the case in which there is a context
# composed of different classes each one with a certain probability
class Ts_learner_context(Learner):

    # n_arms is the number of arms, probabilities is an array of probabilities related to the probability of each single
    # class. User class is an array of indexes each one for each class
    def __init__(self, n_arms, probabilities, user_class, rewards_per_arm=None, collected_rewards=None,
                 beta_parameters=None):
        super().__init__(n_arms)
        self.__probabilities = probabilities
        self.__n_arms = n_arms
        self.__n_classes = len(user_class)
        if rewards_per_arm is None and collected_rewards is None and beta_parameters is None:
            self.__rewards_per_arm = [[[] for arm in range(0, n_arms)] for cls in range(0, self.__n_classes)]
            self.__collected_rewards = [np.array([]) for cls in range(0, self.__n_classes)]
            self.__beta_parameters = [np.ones((n_arms, 2)) for cls in range(0, len(user_class))]
        else:
            self.__rewards_per_arm = rewards_per_arm
            self.__collected_rewards = collected_rewards
            self.__beta_parameters = beta_parameters
        self.__collected_rewards = [np.array([]) for cls in range(0, self.__n_classes)]

    # This method pull every arm and multiply it for the probability of the class. It does so for each class obtaining
    # an array of values for each class (each element = value drawn from the prior distribution of that arm multiplied
    # by the probability of that  class). Then it sum the arrays (with axis=0 !!) and it finds the best arm by computing
    # the argmax of the found array thereby finding the best arm.
    def pull_arm(self):
        scores = np.zeros(self.__n_classes, self.__n_arms)
        for cls in range(0, self.__n_classes):
            scores[cls] = (np.random.beta(self.__beta_parameters[cls][:, 0], self.__beta_parameters[cls][:, 1]) *
                           self.__probabilities[cls])
        best_indexes_for_class = []
        for cls in range(0, self.__n_classes):
            best_indexes_for_class.append(np.argmax(scores[cls]))
        best_indexes_for_class.append(np.argmax(scores.sum(axis=0)))
        return best_indexes_for_class

    # This method updates the distributions. It updates the beta parameter of the optimal arm for each beta of each class
    def update(self, pulled_arm, reward):
        self.t += 1
        self.__update_observations(pulled_arm, reward)
        for cls in range(0, self.__n_classes):
            self.__beta_parameters[cls][pulled_arm, 0] = self.__beta_parameters[cls][pulled_arm, 0] + reward[cls]
            self.__beta_parameters[cls][pulled_arm, 1] = self.__beta_parameters[cls][pulled_arm, 1] + 1.0 - reward[cls]

    # It update the observation like in the basic case but it does so for each class
    def __update_observations(self, pulled_arm, reward):
        for cls in range(0, self.__n_classes):
            self.__rewards_per_arm[cls][pulled_arm].append(reward[cls])
            self.__collected_rewards = np.append(self.__collected_rewards[cls], reward[cls])

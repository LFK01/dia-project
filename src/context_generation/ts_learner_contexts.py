from src.advertising.learner.learner import *


# This class allows to do the usual things of a thompson sampling but in the case in which there is a context
# composed of different classes each one with a certain probability. The class allow the user of split the learner in 2
# learners when it is needed to split the context.
class Ts_learner_context(Learner):

    # n_arms is the number of arms, probabilities is an array of probabilities related to the probability of each single
    # class. User class is an array of indexes each one for each class
    def __init__(self, n_arms, probabilities, user_class, rewards_per_arm=None, collected_rewards=None,
                 beta_parameters=None):
        super().__init__(n_arms)
        self.__probabilities = probabilities
        self.__n_arms = n_arms
        self.__n_classes = len(user_class)
        if rewards_per_arm is None and collected_rewards is None and beta_parameters is None and collected_rewards is None:
            self.__rewards_per_arm = [[[] for arm in range(0, n_arms)] for cls in range(0, self.__n_classes)]
            self.__collected_rewards = [[] for cls in range(0, self.__n_classes)]
            self.__beta_parameters = [np.ones((n_arms, 2)) for cls in range(0, len(user_class))]
        else:
            self.__rewards_per_arm = rewards_per_arm
            self.__collected_rewards = collected_rewards
            self.__beta_parameters = beta_parameters

    # This method pull every arm and multiply it for the probability of the class. It does so for each class obtaining
    # an array of values for each class (each element = value drawn from the prior distribution of that arm multiplied
    # by the probability of that  class). Then it sum the arrays (with axis=0 !!) and it finds the best arm by computing
    # the argmax of the found array thereby finding the best arm. Finally it returns the best arm.
    def pull_arm(self):
        scores = np.zeros((self.__n_classes, self.__n_arms))
        for cls in range(0, self.__n_classes):
            scores[cls] = (np.random.beta(self.__beta_parameters[cls][:, 0], self.__beta_parameters[cls][:, 1]) *
                           self.__probabilities[cls])
        return np.argmax(scores.sum(axis=0))

    # This finds the optimal arm in the case in which we are not considering not all the class as part of a context but
    # only the ones given as input to this method (classes). It finds the best arm considering only "classes" as classes.
    # Classes is an array of indexes. Each index represents a class. Example: classes = [1,3,4] means that in this case
    # we are considering only the classes with those indexes instead of consider all the class which are part of the context
    def get_best_arm_sub_context(self, classes):
        scores = np.zeros((len(classes), self.__n_arms))
        for cls in range(0, len(classes)):
            scores[cls] = (np.random.beta(self.__beta_parameters[classes[cls]][:, 0],
                                          self.__beta_parameters[classes[cls]][:, 1]) *
                           self.__probabilities[classes[cls]])
        index = np.argmax(scores.sum(axis=0))
        return index

    # This method updates the distributions. It updates the beta parameter of the optimal arm for each beta of each class
    def update(self, pulled_arm, reward):
        self.t += 1
        self.__update_observations(pulled_arm, reward)
        for cls in range(0, self.__n_classes):
            self.__beta_parameters[cls][pulled_arm, 0] = self.__beta_parameters[cls][pulled_arm, 0] + reward[cls]
            self.__beta_parameters[cls][pulled_arm, 1] = self.__beta_parameters[cls][pulled_arm, 1] + 1.0 - reward[cls]

    # This method allow to split the learner in two different learners each one for a different context. The learners
    # are given in array form. First classes is an array of classes which compose the first new context and second_classes
    # is an array of classes which compose the second new context.
    def split_in_2(self, first_classes, second_classes):
        probabilities1 = [self.__probabilities[i] for i in first_classes]
        probabilities2 = [self.__probabilities[i] for i in second_classes]
        user_class1 = first_classes
        user_class2 = second_classes
        rewards_per_arm1 = [self.__rewards_per_arm[i] for i in first_classes]
        rewards_per_arm2 = [self.__rewards_per_arm[i] for i in second_classes]
        collected_rewards1 = [self.__collected_rewards[i] for i in first_classes]
        collected_rewards2 = [self.__collected_rewards[i] for i in second_classes]
        beta_parameters1 = [self.__beta_parameters[i] for i in first_classes]
        beta_parameters2 = [self.__beta_parameters[i] for i in second_classes]
        return [self.__init__(self.__n_arms, probabilities1, user_class1, rewards_per_arm1, collected_rewards1,
                              beta_parameters1),
                self.__init__(self.__n_arms, probabilities2, user_class2, rewards_per_arm2, collected_rewards2,
                              beta_parameters2)]

    # It update the observation like in the basic case but it does so for each class which make up the context contained
    # in this learner
    def __update_observations(self, pulled_arm, reward):
        for cls in range(0, self.__n_classes):
            self.__rewards_per_arm[cls][pulled_arm].append(reward[cls])
            self.__collected_rewards[cls].append(reward[cls])

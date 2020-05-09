from src.context_generation.ts_learner_contexts import *
import math as m

# it is the level of confidence
confidence = 0.95


# This is a class that allows the user to run a single iteration of the TS for the context in contained in this class
# and also to split the context in two contexts if and only if it is considered convenient.
class ContextContainer:
    # This is the constructor of the class. It takes as inputs: an array of integer representing the class of which
    # the context represented in this class is composed. It takes  the probabilities of each class which represents
    # the probabilities of a user belonging to a specific class. It takes a list of environments which are the ones
    # of the single classes. Finally it takes the number of arms
    def __init__(self, user_class, context_probabilities, environment, n_arms):
        self.__context = user_class
        self.__probabilities = context_probabilities
        self.__ts_learner_context = Ts_learner_context(n_arms, self.__probabilities, self.__context)
        self.__environment = environment
        self.__n_classes = len(user_class)

    # This method is used for do a single iteration of the thompson sampling algorithm. It does: pull round and update
    # and in the meantime it updates some class parameters used for the computation of the hoeffding bound
    def run_TS(self):
        indexes = self.__ts_learner_context.pull_arm()[-1]
        optimal_arm_index = indexes[-1]
        total_reward = 0
        rewards_this_round = []
        for cls in range(0, self.__n_classes):
            reward_of_class = self.__environment[cls].round(optimal_arm_index) * self.__probabilities[cls]
            rewards_this_round.append(reward_of_class)
            total_reward += reward_of_class
        self.__ts_learner_context.update(optimal_arm_index, rewards_this_round)
        return total_reward

    # Method used for splitting the current context in two different contexts if and only if it is considered convenient
    # by comparing the lower bound multiplied by the probability of it and the sum of the lower bounds of the divided
    # contexts multiplied each one with the related probability
    def split_context(self):
        return

    # Private method used for calculating the hoeffding bound given the classes with which calculate it
    def __compute_hoeffding_bounds(self, the_classes, arm):
        return
import math as m
import numpy as np
from src.assignment_5.ts_learner_contexts import TSLearnerContext
from src.assignment_4.pricing_env import PricingEnv

# it is the level of confidence
confidence = 0.95


# This is a class that allows the user to run a single iteration of the TS for the context in contained in this class
# and also to split the context in two contexts if and only if it is considered convenient.
class ContextContainer:
    # This is the constructor of the class. It takes as inputs: an array of integer representing the class of which
    # the context represented in this class is composed. It takes  the probabilities of each class which represents
    # the probabilities of a user belonging to a specific class. It takes a list of environments which are the ones
    # of the single classes. Finally it takes the number of arms
    def __init__(self, user_class, context_probabilities, environment, n_arms, ts_learner=None):
        self.__context = user_class
        self.__probabilities = context_probabilities
        if ts_learner is None:
            self.__ts_learner_context = TSLearnerContext(n_arms, self.__probabilities, self.__context)
        else:
            self.__ts_learner_context = ts_learner
        self.__environment = environment
        self.__n_classes = len(user_class)
        self.__n_arms = n_arms
        self.__reward_per_arm = [[[] for arm in range(0, self.__n_arms)] for cls in range(0, self.__n_classes)]
        self.__context_optimal_arm = 0

    # This method is used for do a single iteration of the thompson sampling algorithm. It does: pull round and update
    # and in the meantime it updates some class parameters used for the computation of the hoeffding bound
    def run_ts(self):
        self.__context_optimal_arm = self.__ts_learner_context.pull_arm()
        total_reward = 0
        rewards_this_round = []
        for cls in range(0, len(self.__context)):
            reward_of_class = self.__environment[cls].round(self.__context_optimal_arm) * self.__probabilities[cls]
            rewards_this_round.append(reward_of_class / self.__probabilities[cls])
            self.__reward_per_arm[cls][self.__context_optimal_arm].append(reward_of_class / self.__probabilities[cls])
            total_reward += reward_of_class
        self.__ts_learner_context.update(self.__context_optimal_arm, rewards_this_round)
        return total_reward

    # Method used for splitting the current context in two different contexts if and only if it is considered convenient
    # by comparing the lower bound multiplied by the probability of it and the sum of the lower bounds of the divided
    # contexts multiplied each one with the related probability
    def split_context(self):
        # If the number of class in a context is 1, we can't split anymore
        if len(self.__context) == 1:
            raise

        # If we have no enough rewards (data in this case) for some arm in some class, we can't compute the hoeffding bound
        for i in range(0, self.__n_arms):
            for cls in self.__context:
                if len(self.__reward_per_arm[cls][i]) < 4:
                    raise

        current_context_bound = self.__compute_hoeffding_bounds(self.__context, self.__context_optimal_arm)
        possible_splitting = []
        splitting_bounds = []
        splitting_values = []
        for cls in self.__context:
            possible_splitting.append([list(set(self.__context) - {cls}), [cls]])
            optimal_arms = [self.__ts_learner_context.get_best_arm_sub_context(possible_splitting[-1][0]),
                            self.__ts_learner_context.get_best_arm_sub_context(possible_splitting[-1][1])]
            splitting_bounds.append([self.__compute_hoeffding_bounds(possible_splitting[-1][0], optimal_arms[0]),
                                     self.__compute_hoeffding_bounds(possible_splitting[-1][1], optimal_arms[1])])
            splitting_values.append(splitting_bounds[-1][0] + splitting_bounds[-1][1])
        index = int(np.argmax(splitting_values))
        if current_context_bound <= splitting_values[index]:
            userclass1 = possible_splitting[index][0]
            userclass2 = possible_splitting[index][1]
            new_contexts_learners = self.__ts_learner_context.split_in_2(userclass1, userclass2)
            probabilities1 = [self.__probabilities[i] for i in userclass1]
            probabilities2 = [self.__probabilities[i] for i in userclass2]
            environment1 = [self.__environment[i] for i in userclass1]
            environment2 = [self.__environment[i] for i in userclass2]
            # print("splitting1:", userclass1, "splitting 2:", userclass2, "\n")
            print("\nsplit")
            return [ContextContainer(userclass1, probabilities1, environment1, self.__n_arms,
                                     new_contexts_learners[0]), ContextContainer(userclass2, probabilities2,
                                                                                 environment2, self.__n_arms,
                                                                                 new_contexts_learners[1])]
        else:
            raise

    # Private method used for calculating the hoeffding bound given the classes with which calculate it and the optimal
    # arm index
    def __compute_hoeffding_bounds(self, context_classes, optimal_arm_index):
        empirical_mean = 0
        context_probability = 0
        for cls in context_classes:
            empirical_mean += np.mean(self.__reward_per_arm[cls][optimal_arm_index])
            context_probability += self.__probabilities[cls]
        return context_probability * (empirical_mean - m.sqrt(
            -m.log10(1 - confidence) / 2 * len(
                self.__reward_per_arm[context_classes[0]][optimal_arm_index])))

    def print_context(self, context_id):
        print("Context ", context_id, ": ", self.__context)



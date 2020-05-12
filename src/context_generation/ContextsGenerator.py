import numpy as np
import matplotlib.pyplot as plt

from src.context_generation.ContextContainer import *
from src.pricing.reward_function import rewards
from src.pricing.environment import *


class ContextsGenerator:
    def __init__(self, user_class, user_class_probabilities, environment, n_arms):
        self.contexts = [ContextContainer(user_class, user_class_probabilities, environment, n_arms)]
        self.rewards = []
        self.opt = 0
        for i in range(0, 3):
            self.opt += user_class_probabilities[i] * np.amax(
                [environment[i].round(arm) for arm in range(0, environment[i].n_arms)])

    # Create a new context trying to split each of the context
    def generate_new_context(self):
        for i in range(0, len(self.contexts)):
            try:
                split = self.contexts[i].split_context()
                self.contexts.pop(i)  # Remove the current context that has been splitted
                for s in split:
                    self.contexts.append(s)  # Insert the new created context
                print(len(self.contexts))
                break
            except:
                continue

    # Call the ts algorithm for each of the context
    def run_ts(self):
        total_rewards = 0
        for c in self.contexts:
            total_rewards += c.run_TS()

        self.rewards.append(total_rewards)


def optimal_for_partition(classes_of_partition, all_probabilities, all_rewards):
    scores = np.array([all_rewards[cls] * all_probabilities[cls] for cls in classes_of_partition])
    return np.max(scores.sum(axis=0))


def compute_optimum(all_classes, all_probabilities, all_rewards):
    values_of_partitions = []
    partitions = []
    for cls in all_classes:
        partition1 = list(set(all_classes) - {cls})
        partition2 = [cls]
        partitions.append([partition1, partition2])
        values_of_partitions.append(
            optimal_for_partition(partition1, all_probabilities, all_rewards) + optimal_for_partition(partition2,
                                                                                                      all_probabilities,
                                                                                                      all_rewards))
    value_all_splitted = 0
    for cls in all_classes:
        value_all_splitted += optimal_for_partition([cls], all_probabilities, all_rewards)
    values_of_partitions.append(value_all_splitted)
    index = int(np.argmax(values_of_partitions))
    print("optimal partition: ", partitions[index], "\n")
    return values_of_partitions[index]


if __name__ == '__main__':
    T = 1400
    n_experiment = 400

    n_arms = 11
    min_price = 0.0
    max_price = 1.0
    prices = np.linspace(min_price, max_price, n_arms)

    # Create a reward curve for each class
    rewards = [rewards(prices, max_price) for i in range(0, 3)]
    # Create an environment for each class
    environment = [Environment(n_arms=n_arms, probabilities=rewards[cls]) for cls in range(0, 3)]

    ts_rewards_per_experiment = []

    user_class = [0, 1, 2]
    user_class_probabilities = [0.3, 0.3, 0.4]

    for e in range(0, n_experiment):
        context_generator = ContextsGenerator(user_class=user_class, user_class_probabilities=user_class_probabilities,
                                              environment=environment, n_arms=n_arms)
        for t in range(0, T):
            # Every 7 days try the new context generation
            if (t + 1) % 7 == 0:
                context_generator.generate_new_context()
            context_generator.run_ts()

        # Collect the rewards for each experiment
        ts_rewards_per_experiment.append(context_generator.rewards)

    # TODO Calcolare il regret tramite opt e ts_rewards_per_experiment
    ts_instantaneous_regret = compute_optimum(user_class, user_class_probabilities, rewards) - np.mean(ts_rewards_per_experiment, axis=0)
    # plot the results
    print("Opt: ")
    np.set_printoptions(precision=3)
    print(compute_optimum(user_class, user_class_probabilities, rewards))
    print("Rewards")
    np.set_printoptions(precision=3)
    print(ts_rewards_per_experiment)
    print("Regret")
    np.set_printoptions(precision=3)
    print(ts_instantaneous_regret)
    plt.figure(0)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'g')
    plt.plot(context_generator.opt, '--k')
    plt.legend(["TS", "Optimum"])
    plt.show()

    plt.figure(1)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(ts_instantaneous_regret), 'g')
    plt.legend(["TS"])
    plt.show()

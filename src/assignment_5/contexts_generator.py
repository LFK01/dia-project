import numpy as np
from src.assignment_5.context_container import ContextContainer


class ContextsGenerator:
    def __init__(self, user_class, user_class_probabilities, environment, n_arms):
        self.contexts = [ContextContainer(user_class, user_class_probabilities, environment, n_arms)]
        self.rewards = []
        self.opt = 0

    # Create a new context trying to split each of the context
    def generate_new_context(self):
        for i in range(0, len(self.contexts)):
            try:
                split = self.contexts[i].split_context()
                self.contexts.pop(i)  # Remove the current context that has been split
                for s in split:
                    self.contexts.append(s)  # Insert the new created context
                break
            except:
                continue

    # Call the ts algorithm for each of the context
    def run_ts(self):
        total_rewards = 0
        for c in self.contexts:
            total_rewards += c.run_ts()
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

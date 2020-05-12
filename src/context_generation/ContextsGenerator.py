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


if __name__ == '__main__':
    T = 600
    n_experiment = 50

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
    user_class_probabilities = [0.2, 0.5, 0.3]

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
    ts_instantaneous_regret = context_generator.opt - np.mean(ts_rewards_per_experiment, axis=0)
    # plot the results
    print("Opt: ")
    np.set_printoptions(precision=3)
    print(context_generator.opt)
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

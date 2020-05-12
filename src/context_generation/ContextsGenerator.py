import numpy as np
from src.context_generation.ContextContainer import *
from src.pricing.reward_function import rewards
from src.pricing.environment import *


class ContextsGenerator:
    def __init__(self, user_class, user_class_probabilities, environment):
        self.contexts = [ContextContainer(user_class, user_class_probabilities, environment)]
        self.rewards = []
        self.opt = 0
        for i in range(0, 3):
            self.opt += np.max(environment[i].round(arm) for arm in range(0, environment[i].n_arms))

    # Metodo per ricreare un nuovo contesto
    def generate_new_context(self):
        new_context = []
        for c in self.contexts:
            new_context.append(c.split_context())

        self.contexts = [new_context]

    # chiama il TS in ogni contesto
    def run_ts(self):
        total_rewards = 0
        for c in self.contexts:
            total_rewards += c.run_TS()
            
        self.rewards.append(total_rewards)


T = 100
n_experiment = 10

n_arms = 11
min_price = 0.0
max_price = 1.0
prices = np.linspace(min_price, max_price, n_arms)

rewards = [rewards(prices) for i in range(0, 3)]
environment = [Environment(n_arms=n_arms, probabilities=rewards) for i in range(0, 3)]

ts_rewards_per_experiment = []

user_class = [0, 1, 2]
user_class_probabilities = [0.2, 0.5, 0.3]

for e in range(0, n_experiment):
    opt_per_round = 0
    context_generator = ContextsGenerator(user_class=user_class, user_class_probabilities=user_class_probabilities,
                                          environment=environment)

    for t in range(0, T):
        if (t + 1) % 7 == 0:
            context_generator.generate_new_context()
        context_generator.run_ts()

    ts_rewards_per_experiment.append(context_generator.rewards)

# TODO Calcolare il regret tramite opt e ts_rewards_per_experiment

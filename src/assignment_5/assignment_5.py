import numpy as np
import matplotlib.pyplot as plt
from src.assignment_4.reward_function import rewards
from src.assignment_4.pricing_env import PricingEnv
from src.assignment_5.contexts_generator import ContextsGenerator
from tqdm import tqdm

T = 18250
n_experiment = 100

n_arms = 11
min_price = 0.0
max_price = 1.0
prices = np.linspace(min_price, max_price, n_arms)

# Create a reward curve for each class
rewards = [rewards(prices, max_price) for i in range(0, 3)]
# Create an environment for each class
environment = [PricingEnv(n_arms=n_arms, conversion_rates=rewards[cls]) for cls in range(0, 3)]

ts_rewards_per_experiment = []
opt_per_experiment = []

user_class = [0, 1, 2]
user_class_probabilities = [0.1, 0.5, 0.4]

for e in tqdm(range(0, n_experiment), desc="Experiment processed", unit="exp"):
    context_generator = ContextsGenerator(user_class=user_class, user_class_probabilities=user_class_probabilities,
                                          environment=environment, n_arms=n_arms)
    for t in range(0, T):
        # Every 7 days try the new context generation
        if (t + 1) % 350 == 0:
            context_generator.generate_new_context()
        context_generator.run_ts()

    # print("Experiment ", e)
    # for contextId in range(0, len(context_generator.contexts)):
    #     context_generator.contexts[contextId].print_context(contextId)

    # Collect the rewards for each experiment
    ts_rewards_per_experiment.append(context_generator.rewards)
    opt_per_experiment.append(context_generator.opt)

# TODO Calcolare il regret tramite opt e ts_rewards_per_experiment
# ts_instantaneous_regret = compute_optimum(user_class, user_class_probabilities, rewards) - np.mean(
#     ts_rewards_per_experiment, axis=0)
ts_instantaneous_regret = np.mean(np.array(opt_per_experiment) -
                                  np.array(ts_rewards_per_experiment), axis=0)
# plot the results
print("Opt: ")
np.set_printoptions(precision=3)
# print(compute_optimum(user_class, user_class_probabilities, rewards))
print(np.mean(opt_per_experiment, axis=0))
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
plt.plot(np.mean(opt_per_experiment, axis=0), 'r')
plt.legend(["TS", "Optimum"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(ts_instantaneous_regret), 'g')
plt.legend(["TS"])
plt.show()

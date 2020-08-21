import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.assignment_4.reward_function2 import rewards
from src.assignment_4.pricing_env import PricingEnv
from src.assignment_5.contexts_generator import ContextsGenerator, compute_optimum
from tqdm import tqdm
from scipy import interpolate

T = 18250
n_experiment = 100
min_price = 0.0
max_price = 1.00

user_class = [0, 1, 2]
user_class_probabilities = [0.1, 0.5, 0.4]

readFile = '../data/pricing.csv'

# Read environment data from csv file
data = pd.read_csv(readFile)
n_arms = len(data.columns)

x_values = [np.linspace(min_price, max_price, n_arms) for i in range(0, len(user_class))]
y_values = []
# The values of the y for each function
for i in range(0, len(data.index)):
    y_values.append(np.array(data.iloc[i]))

prices = np.linspace(min_price, max_price, n_arms)

demand_functions = [interpolate.interp1d(x_values[i], y_values[i]) for i in user_class]

# Create a reward curve for each class
rewards = [rewards(prices, demand_functions[i], i + 1) for i in range(0, 3)]
# Create an environment for each class
environment = [PricingEnv(n_arms=n_arms, conversion_rates=rewards[cls]) for cls in range(0, 3)]

ts_rewards_per_experiment = []
# opt_per_experiment = []


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

# TODO Calcolare il regret tramite opt e ts_rewards_per_experiment
opt = compute_optimum(user_class, user_class_probabilities, rewards)
print("opt: ", opt, "\n")
ts_instantaneous_regret = opt - np.mean(
    ts_rewards_per_experiment, axis=0)
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
plt.plot(opt, '--k')
plt.legend(["TS", "Optimum"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(ts_instantaneous_regret), 'g')
plt.legend(["TS"])
plt.show()

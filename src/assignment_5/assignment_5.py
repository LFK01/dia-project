import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.assignment_4.reward_function import rewards
from src.assignment_4.pricing_env import PricingEnv
from src.assignment_5.contexts_generator import ContextsGenerator, compute_optimum
from tqdm import tqdm
from scipy import interpolate

from src.utils.constants import img_path

T = 18250
n_experiment = 50
min_price = 0.0
max_price = 100.0

user_class = [0, 1, 2]
user_class_probabilities = [2 / 10, 2 / 5, 2 / 5]

readFile = '../data/pricing.csv'

# Read environment data from csv file
data = pd.read_csv(readFile)
n_arms = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

y_values = []
# The values of the y for each function
for i in range(0, len(data.index)):
    y_values.append(np.array(data.iloc[i]))
x_values = [np.linspace(min_price, max_price, len(y_values[u])) for u in user_class]

prices = np.linspace(min_price, max_price, n_arms)

demand_functions = [interpolate.interp1d(x_values[i], y_values[i]) for i in user_class]

# Create a reward curve for each class
rewards = [rewards(prices, demand_functions[i], i + 1) for i in range(0, 3)]
# Create an environment for each class
environment = [PricingEnv(n_arms=n_arms, conversion_rates=demand_functions[cls](prices)) for cls in user_class]

ts_rewards_per_experiment = []
# opt_per_experiment = []


for e in tqdm(range(0, n_experiment), desc="Experiment processed", unit="exp"):
    context_generator = ContextsGenerator(user_class=user_class, user_class_probabilities=user_class_probabilities,
                                          environment=environment, n_arms=n_arms, prices=prices)
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
ts_instantaneous_regret = opt - ts_rewards_per_experiment
# plot the results
print("Opt: ")
np.set_printoptions(precision=3)
print(compute_optimum(user_class, user_class_probabilities, rewards))
print("Rewards")
np.set_printoptions(precision=3)
# print(np.mean(ts_rewards_per_experiment, axis=0) * 100)
print("Regret")
np.set_printoptions(precision=3)
print(np.mean(ts_instantaneous_regret, axis=0))
plt.figure(0)
plt.ylabel("Cumulative Reward")
plt.xlabel("t")
ts_total_rew = np.cumsum(np.mean(ts_rewards_per_experiment, axis=0))
plt.plot(ts_total_rew, 'g')
plt.plot(opt, '--k')
plt.scatter(len(ts_total_rew), round(np.max(ts_total_rew), 2))
plt.annotate(round(np.max(ts_total_rew), 2), (len(ts_total_rew), round(np.max(ts_total_rew), 2)),
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc3"),
             xytext=(len(ts_total_rew) - 3000, np.max(ts_total_rew) - 150000))
plt.legend(["TS"])
img_name = "assignment_5_reward.png"
plt.savefig(os.path.join(img_path, img_name))
plt.show()

plt.figure(1)
plt.ylabel("Cumulative Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(np.array(opt) - ts_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS"])
img_name = "assignment_5_cum_regret.png"
plt.savefig(os.path.join(img_path, img_name))
plt.show()

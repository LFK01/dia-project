import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from tqdm import tqdm

from src.assignment_4.greedy_learner import GreedyLearner
from src.assignment_4.reward_function import rewards
from src.assignment_4.ts_learner import TSLearner
from src.assignment_4.pricing_env import PricingEnv
from src.utils.constants import subcampaign_names, img_path

T = 18250

n_experiments = 50

min_price = 0.0
max_price = 100.0
n_arms = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

subcampaigns = [0, 1, 2]
readFile = '../data/pricing.csv'

# Read environment data from csv file
data = pd.read_csv(readFile)
# n_arms = 10

y_values = []
# The values of the y for each function
for i in range(0, len(data.index)):
    y_values.append(np.array(data.iloc[i]))
x_values = [np.linspace(min_price, max_price, len(y_values[s])) for s in subcampaigns]

probabilities_vector = [2 / 10, 2 / 5, 2 / 5]
conversion_prices = np.linspace(min_price, max_price, n_arms)
all_rewards_vector = []

demand_functions = [interpolate.interp1d(x_values[i], y_values[i]) for i in subcampaigns]

for subcampaign in subcampaigns:
    plt.ylabel("Conversion Probability")
    plt.xlabel("Prices")
    x = np.linspace(min_price, max_price, 100)
    y = [demand_functions[subcampaign](x[i]) for i in range(0, 100)]
    plt.plot(x, y, 'g')
    plt.legend(["Subcampaign " + str(subcampaign + 1) + " " + subcampaign_names[subcampaign]])
    img_name = "demand_curve_" + str(subcampaign + 1) + ".png"
    plt.savefig(os.path.join(img_path, img_name))
    plt.show()

rewards = [rewards(conversion_prices, demand_functions[i], i + 1) for i in subcampaigns]
aggregated_rewards = []
for arm in range(0, n_arms):
    gain = 0
    for campaign in subcampaigns:
        gain += probabilities_vector[campaign] * rewards[campaign][arm]
    aggregated_rewards.append(gain)

opt = np.max(aggregated_rewards)
print(opt)

environments = []

ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in tqdm(range(0, n_experiments), desc="Experiment processed", unit="exp"):

    ts_learner = TSLearner(n_arms=n_arms,
                           probabilities=probabilities_vector,
                           number_of_classes=len(probabilities_vector), prices=conversion_prices)
    gr_learner = GreedyLearner(n_arms=n_arms,
                               probabilities=probabilities_vector,
                               number_of_classes=len(probabilities_vector), prices=conversion_prices)

    for subcampaign in range(len(subcampaigns)):
        environments.append(PricingEnv(n_arms, demand_functions[subcampaign](conversion_prices)))

    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()

        reward_per_round = []
        for subcampaign in range(len(subcampaigns)):
            reward = environments[subcampaign].round(pulled_arm) * conversion_prices[pulled_arm]
            reward_per_round.append(reward)

        ts_learner.update(pulled_arm, reward_per_round)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()

        reward_per_round = []

        for subcampaign in range(len(subcampaigns)):
            reward_per_round.append(environments[subcampaign].round(pulled_arm) * conversion_prices[pulled_arm])
        gr_learner.update(pulled_arm, reward_per_round)

    ts_rewards_per_experiment.append(
        np.average(a=ts_learner.collected_rewards, axis=0, weights=probabilities_vector))
    gr_rewards_per_experiment.append(
        np.average(a=gr_learner.collected_rewards, axis=0, weights=probabilities_vector))

plt.figure(0)
plt.ylabel("Cumulative Reward")
plt.xlabel("t")
print(ts_rewards_per_experiment)
ts_total_rew = np.cumsum(np.mean(ts_rewards_per_experiment, axis=0))
gr_total_rew = np.cumsum(np.mean(gr_rewards_per_experiment, axis=0))
print(ts_total_rew)
plt.plot(ts_total_rew, 'g')
plt.scatter(len(ts_total_rew), round(np.max(ts_total_rew), 2))
plt.annotate(round(np.max(ts_total_rew), 2), (len(ts_total_rew), round(np.max(ts_total_rew), 2)),
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc3"),
             xytext=(len(ts_total_rew) - 5000, np.max(ts_total_rew)))
plt.plot(gr_total_rew, 'r')
plt.legend(["TS", "Greedy"])

img_name = "assignment_4_rewards.png"
plt.savefig(os.path.join(img_path, img_name))
plt.show()

plt.figure(1)
plt.ylabel("Cumulative Regret")
plt.xlabel("t")
ts_total_reg = np.cumsum(np.mean(np.array(opt) - ts_rewards_per_experiment, axis=0))
gr_total_reg = np.cumsum(np.mean(np.array(opt) - gr_rewards_per_experiment, axis=0))
print(np.mean(np.array(opt) - ts_rewards_per_experiment, axis=0))
print(ts_total_reg)
plt.plot(ts_total_reg, 'g')
plt.plot(gr_total_reg, 'r')
plt.legend(["TS", "Greedy"])

img_name = "assignment_4_regrets.png"
plt.savefig(os.path.join(img_path, img_name))
plt.show()

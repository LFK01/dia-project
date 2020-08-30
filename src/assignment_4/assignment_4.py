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

T = 300

n_experiments = 10

min_price = 0.0
max_price = 100.0
# n_arms = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

subcampaigns = [0, 1, 2]
readFile = '../data/pricing.csv'

# Read environment data from csv file
data = pd.read_csv(readFile)
n_arms = 10

y_values = []
# The values of the y for each function
for i in range(0, len(data.index)):
    y_values.append(np.array(data.iloc[i]))
x_values = [np.linspace(min_price, max_price, len(y_values[s])) for s in subcampaigns]

probabilities_vector = [1 / 4, 1 / 2, 1 / 4]
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

rewards_normalized = []
opt = np.max(aggregated_rewards)
print(opt)
opt_normalized = np.divide(opt, max_price)
print(opt_normalized)
opt_per_campaign = [np.max(rewards[i]) for i in subcampaigns]

for campaign in subcampaigns:
    rewards_normalized.append(np.divide(rewards[campaign], max_price))
environments = []

ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

# for subcampaign in range(len(subcampaigns)):
# # ts_rewards_per_experiment.append([])
# # gr_rewards_per_experiment.append([])

for e in tqdm(range(0, n_experiments), desc="Experiment processed", unit="exp"):

    ts_learner = TSLearner(n_arms=n_arms,
                           probabilities=probabilities_vector,
                           number_of_classes=len(probabilities_vector))
    gr_learner = GreedyLearner(n_arms=n_arms,
                               probabilities=probabilities_vector,
                               number_of_classes=len(probabilities_vector))

    for subcampaign in range(len(subcampaigns)):
        environments.append(PricingEnv(n_arms, rewards_normalized[subcampaign]))

    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()

        reward_per_round = []

        for subcampaign in range(len(subcampaigns)):
            reward_per_round.append(environments[subcampaign].round(pulled_arm))

        ts_learner.update(pulled_arm, reward_per_round)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()

        reward_per_round = []

        for subcampaign in range(len(subcampaigns)):
            reward_per_round.append(environments[subcampaign].round(pulled_arm))
        gr_learner.update(pulled_arm, reward_per_round)

    ts_rewards_per_experiment.append(
        np.average(a=ts_learner.collected_rewards, axis=0, weights=probabilities_vector))
    gr_rewards_per_experiment.append(
        np.average(a=gr_learner.collected_rewards, axis=0, weights=probabilities_vector))
    print(ts_rewards_per_experiment)

# fig, axs = plt.subplots(3, 2, figsize=(14, 8))
# for subcampaign in range(len(subcampaigns)):
#     # axs[subcampaign, 0].figure("subcampaign" + str(subcampaign) + ".1")
#     axs[subcampaign, 0].plot(
#         np.cumsum(np.mean(np.array(opt_normalized) - ts_rewards_per_experiment[subcampaign], axis=0)), 'r')
#     axs[subcampaign, 0].plot(
#         np.cumsum(np.mean(np.array(opt_normalized) - gr_rewards_per_experiment[subcampaign], axis=0)), 'g')
#     axs[subcampaign, 0].legend(["TS", "Greedy"])
#
#     # axs.figure("subcampaign" + str(subcampaign) + ".2")
#     axs[subcampaign, 1].plot((np.mean(np.array(opt_normalized) - ts_rewards_per_experiment[subcampaign], axis=0)), 'r')
#     axs[subcampaign, 1].plot((np.mean(np.array(opt_normalized) - gr_rewards_per_experiment[subcampaign], axis=0)), 'g')
#     axs[subcampaign, 1].legend(["TS", "Greedy"])
#
# for ax in axs.flat:
#     if list(axs.flat).index(ax) % 2 == 0:
#         ax.set(xlabel='t', ylabel='CumRegret')
#     else:
#         ax.set(xlabel='t', ylabel='Regret')
# ax.label_outer()

# plt.figure(0)
# plt.ylabel("Reward")
# plt.xlabel("t")
# print(ts_rewards_per_experiment[0])
# total_rew = []
# for i in range(0, len(np.cumsum(np.mean(ts_rewards_per_experiment[0], axis=0)))):
#     total_rew.append(0)
# for s in subcampaigns:
#     total_rew += np.cumsum(np.mean(ts_rewards_per_experiment[s], axis=0))
# print(total_rew)
# plt.plot(total_rew, 'g')
# plt.plot(opt, '--k')
# plt.legend(["TS", "Optimum"])
#
# img_name = "assignment_4_rewards.png"
# plt.savefig(os.path.join(img_path, img_name))
# plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
total_reg = np.cumsum(np.mean(np.array(opt_normalized) - ts_rewards_per_experiment, axis=0))
# for i in range(0, len(np.cumsum(np.mean(np.array(opt) - ts_rewards_per_experiment[0], axis=0)))):
#     total_reg.append(0)
# for s in subcampaigns:
#     total_reg += np.cumsum(np.mean(np.array(opt) - ts_rewards_per_experiment[s], axis=0) * probabilities_vector[s])
print(total_reg)
plt.plot(total_reg, 'g')
plt.legend(["TS"])

img_name = "assignment_4_regrets.png"
plt.savefig(os.path.join(img_path, img_name))
plt.show()

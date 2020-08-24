import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

from src.assignment_4.greedy_learner import GreedyLearner
from src.assignment_4.reward_function import rewards
from src.assignment_4.ts_learner import TSLearner
from src.assignment_4.pricing_env import PricingEnv

T = 300

n_experiments = 100

min_price = 0.0
max_price = 100.0
# n_arms = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

subcampaigns = [0, 1, 2]
readFile = '../data/pricing.csv'

# Read environment data from csv file
data = pd.read_csv(readFile)
n_arms = len(data.columns)

x_values = [np.linspace(min_price, max_price, n_arms) for i in range(0, len(subcampaigns))]
y_values = []
# The values of the y for each function
for i in range(0, len(data.index)):
    y_values.append(np.array(data.iloc[i]))

probabilities_vector = [1 / 4, 1 / 2, 1 / 4]

conversion_prices = np.linspace(min_price, max_price, n_arms)
all_rewards_vector = []

demand_functions = [interpolate.interp1d(x_values[i], y_values[i]) for i in subcampaigns]
for subcampaign in subcampaigns:
    plt.ylabel("Conversion_probability")
    plt.xlabel("Prices")
    x = np.linspace(min_price, max_price, 100)
    y = [demand_functions[subcampaign](x[i]) for i in range(0, 100)]
    plt.plot(x, y, 'g')
    plt.legend(["demand function of the subcampaign " + str(subcampaign + 1)])
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
opt_normalized = np.divide(opt, opt)
opt_per_campaign = [np.max(rewards[i]) for i in subcampaigns]

for campaign in subcampaigns:
    rewards_normalized.append(np.divide(rewards[campaign], opt_per_campaign[campaign]))
environments = []

ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for subcampaign in range(len(subcampaigns)):
    ts_rewards_per_experiment.append([])
    gr_rewards_per_experiment.append([])

for e in range(0, n_experiments):

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

    for subcampaign in range(len(subcampaigns)):
        ts_rewards_per_experiment[subcampaign] \
            .append(np.average(a=ts_learner.collected_rewards, axis=0, weights=probabilities_vector))
        gr_rewards_per_experiment[subcampaign] \
            .append(np.average(a=gr_learner.collected_rewards, axis=0, weights=probabilities_vector))

fig, axs = plt.subplots(3, 2, figsize=(14, 8))
for subcampaign in range(len(subcampaigns)):
    # axs[subcampaign, 0].figure("subcampaign" + str(subcampaign) + ".1")
    axs[subcampaign, 0].plot(
        np.cumsum(np.mean(np.array(opt_normalized) - ts_rewards_per_experiment[subcampaign], axis=0)), 'r')
    axs[subcampaign, 0].plot(
        np.cumsum(np.mean(np.array(opt_normalized) - gr_rewards_per_experiment[subcampaign], axis=0)), 'g')
    axs[subcampaign, 0].legend(["TS", "Greedy"])

    # axs.figure("subcampaign" + str(subcampaign) + ".2")
    axs[subcampaign, 1].plot((np.mean(np.array(opt_normalized) - ts_rewards_per_experiment[subcampaign], axis=0)), 'r')
    axs[subcampaign, 1].plot((np.mean(np.array(opt_normalized) - gr_rewards_per_experiment[subcampaign], axis=0)), 'g')
    axs[subcampaign, 1].legend(["TS", "Greedy"])

for ax in axs.flat:
    if list(axs.flat).index(ax) % 2 == 0:
        ax.set(xlabel='t', ylabel='CumRegret')
    else:
        ax.set(xlabel='t', ylabel='Regret')
    # ax.label_outer()

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from src.assignment_4.greedy_learner import GreedyLearner
from src.assignment_4.reward_function2 import rewards
from src.assignment_4.ts_learner import TSLearner
from src.assignment_4.pricing_env import PricingEnv

T = 300

n_experiments = 100

min_price = 0.0
max_price = 100.0

# n_arms = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))
n_arms = 15

subcampaigns = [0, 1, 2]

probabilities_vector = [1 / 4, 1 / 2, 1 / 4]

conversion_prices = np.linspace(min_price, max_price, n_arms)
all_rewards_vector = []

x_values = [np.linspace(min_price, max_price, 21) for i in range(0, len(subcampaigns))]
# The values of the y for each function
y_values = [np.array([1, 1, 0.99, 0.97, 0.94, 0.90, 0.85, 0.79, 0.72, 0.63, 0.52, 0.39, 0.26, 0.16, 0.08, 0.04, 0.02, 0.02, 0.01, 0, 0]),
            np.array([1, 1, 1, 0.985, 0.955, 0.91, 0.85, 0.775, 0.685, 0.58, 0.46, 0.34, 0.23, 0.14, 0.07, 0.02, 0.01, 0.005, 0.004, 0.002, 0]),
            np.array([1, 1, 1, 1, 0.98, 0.94, 0.86, 0.70, 0.50, 0.35, 0.25, 0.20, 0.17, 0.15, 0.13, 0.11, 0.08, 0.07, 0.04, 0.015, 0])]
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

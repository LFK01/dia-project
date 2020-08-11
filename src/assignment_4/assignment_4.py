import numpy as np
import matplotlib.pyplot as plt

from src.assignment_4.greedy_learner import GreedyLearner
from src.assignment_4.reward_function import rewards
from src.assignment_4.ts_learner import TSLearner
from src.assignment_4.pricing_env import PricingEnv

T = 100

n_experiments = 100

min_price = 0.0
max_price = 100.0

n_arms = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

subcampaigns = [0, 1, 2]

probabilities_vector = [1 / 4, 1 / 2, 1 / 4]

conversion_prices = np.linspace(min_price, max_price, n_arms)
all_rewards_vector = []

rewards = rewards(conversion_prices, max_price)

opt = np.max(rewards)
rewards_normalized = np.divide(rewards, opt)
opt_normalized = np.divide(opt, opt)
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
        environments.append(PricingEnv(n_arms, rewards_normalized))

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
        ts_rewards_per_experiment[subcampaign]\
            .append(np.average(a=ts_learner.collected_rewards, axis=0, weights=probabilities_vector))
        gr_rewards_per_experiment[subcampaign]\
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

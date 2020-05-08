import matplotlib.pyplot as plt
from src.pricing.environment import *
from src.pricing.greedy_learner import *
from src.pricing.ts_learner import *
from src.pricing.reward_function import rewards

T = 100

n_experiments = 100

min_price = 0.0
max_price = 100.0

n_arms = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

subcampaigns = [0, 1, 2]
conversion_prices = np.linspace(min_price, max_price, n_arms)
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

    gr_learners = []
    ts_learners = []

    for subcampaign in range(len(subcampaigns)):
        environments.append(Environment(n_arms, rewards_normalized))
        ts_learners.append(TSLearner(n_arms=n_arms))
        gr_learners.append(GreedyLearner(n_arms=n_arms))
    for t in range(0, T):
        # Thompson Sampling Learner
        for subcampaign in range(len(subcampaigns)):
            pulled_arm = ts_learners[subcampaign].pull_arm()
            reward = environments[subcampaign].round(pulled_arm)
            ts_learners[subcampaign].update(pulled_arm, reward)

        # Greedy Learner
        for subcampaign in range(len(subcampaigns)):
            pulled_arm = gr_learners[subcampaign].pull_arm()
            reward = environments[subcampaign].round(pulled_arm)
            gr_learners[subcampaign].update(pulled_arm, reward)

    for subcampaign in range(len(subcampaigns)):
        ts_rewards_per_experiment[subcampaign].append(ts_learners[subcampaign].collected_rewards)
        gr_rewards_per_experiment[subcampaign].append(gr_learners[subcampaign].collected_rewards)

fig, axs = plt.subplots(3, 2, figsize=(14, 8))
for subcampaign in range(len(subcampaigns)):
    # axs[subcampaign, 0].figure("subcampaign" + str(subcampaign) + ".1")
    axs[subcampaign, 0].plot(np.cumsum(np.mean(np.array(opt_normalized) - ts_rewards_per_experiment[subcampaign], axis=0)), 'r')
    axs[subcampaign, 0].plot(np.cumsum(np.mean(np.array(opt_normalized) - gr_rewards_per_experiment[subcampaign], axis=0)), 'g')
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

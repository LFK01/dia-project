import matplotlib.pyplot as plt
from tqdm import tqdm

from src.advertising.learner.gpts_learner import GPTSLearner
from src.advertising.solver.knapsack import Knapsack
from src.pricing.environment import *
from src.pricing.ts_learner import *
from src.pricing.reward_function import rewards

T = 100

n_experiments = 100

min_budget = 0.0
max_budget = 1.0

n_arms = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

daily_budget = np.linspace(min_budget, max_budget, n_arms)

subcampaigns = [0, 1, 2]
conversion_prices = np.linspace(min_budget, max_budget, n_arms)
rewards = rewards(conversion_prices)
n_arms = len(rewards)
opt = np.max(rewards)

environments = []

ts_rewards_per_experiment = []

for subcampaign in range(len(subcampaigns)):
    ts_rewards_per_experiment.append([])

for e in tqdm(range(0, n_experiments), desc="Experiment processed", unit="exp"):

    ts_learners = []
    gpts_learner = []

    total_clicks_per_t = []

    for s in subcampaigns:
        environments.append(Environment(n_arms=n_arms, probabilities=rewards))
        ts_learners.append(TSLearner(n_arms=n_arms))
        gpts_learner.append(GPTSLearner(n_arms=n_arms, arms=daily_budget))
        # add gp learner

    for t in range(0, T):

        values_combination_of_each_subcampaign = []

        # Thompson Sampling and GP-TS Learner
        for s in subcampaigns:
            pulled_arm = ts_learners[s].pull_arm()
            reward = environments[s].round(pulled_arm)
            ts_learners[s].update(pulled_arm, reward)

            values_combination_of_each_subcampaign.\
                append(gpts_learner[s].pull_arm())

        # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
        # and save the chosen arm of each sub campaign

        superarm = Knapsack(values_combination_of_each_subcampaign, daily_budget).solve()

        # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
        total_clicks = 0
        for s in subcampaigns:
            reward = environments[s].round(superarm[s])
            total_clicks += reward
            gpts_learner[s].update(superarm[s], reward)

        total_clicks_per_t.append(total_clicks)

    for s in subcampaigns:
        ts_rewards_per_experiment[s].append(ts_learners[s].collected_rewards)

fig, axs = plt.subplots(3, 2, figsize=(14, 8))
for subcampaign in range(len(subcampaigns)):
    # axs[subcampaign, 0].figure("subcampaign" + str(subcampaign) + ".1")
    axs[subcampaign, 0].plot(np.cumsum(np.mean(np.array(opt) - ts_rewards_per_experiment[subcampaign], axis=0)), 'r')
    axs[subcampaign, 0].legend(["TS", "Greedy"])

    # axs.figure("subcampaign" + str(subcampaign) + ".2")
    axs[subcampaign, 1].plot((np.mean(np.array(opt) - ts_rewards_per_experiment[subcampaign], axis=0)), 'r')
    axs[subcampaign, 1].legend(["TS", "Greedy"])

for ax in axs.flat:
    if list(axs.flat).index(ax) % 2 == 0:
        ax.set(xlabel='t', ylabel='CumRegret')
    else:
        ax.set(xlabel='t', ylabel='Regret')
    # ax.label_outer()

plt.show()

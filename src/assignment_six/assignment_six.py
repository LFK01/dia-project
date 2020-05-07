import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from src.advertising.learner.gpts_learner import GPTSLearner
from src.advertising.solver.knapsack import Knapsack
from src.assignment_six.advanced_ts_learner import AdvancedTSLearner
from src.pricing.environment import Environment as PricingEnvironment
from src.advertising.environment.click_budget import ClickBudget as AdvertisingEnvironment
from src.pricing.reward_function import rewards

T = 100

n_experiments = 5

subcampaigns = [0, 1, 2]

min_budget_advertising = 0.0
max_budget_advertising = 1.0
sigma_advertising = 10

n_arms_advertising = 21

daily_budget = np.linspace(min_budget_advertising, max_budget_advertising, n_arms_advertising)

min_price_pricing = 0.0
max_price_pricing = 100.0

n_arms_pricing = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

conversion_prices = np.linspace(min_price_pricing, max_price_pricing, n_arms_pricing)
rewards = rewards(conversion_prices, max_price_pricing)
opt = np.max(rewards)
rewards_normalized = np.divide(rewards, opt)
opt_normalized = np.divide(opt, opt)

environments_pricing = []
environments_advertising = []

ts_rewards_per_experiment = []
gp_rewards_per_experiment = []

for subcampaign in range(len(subcampaigns)):
    ts_rewards_per_experiment.append([])
    gp_rewards_per_experiment.append([])

for e in tqdm(range(0, n_experiments), desc="Experiment processed", unit="exp"):

    advanced_ts_learners = []
    gpts_learner = []

    total_clicks_per_t = []

    for s in subcampaigns:
        environments_pricing.append(PricingEnvironment(n_arms=n_arms_pricing, probabilities=rewards_normalized))
        environments_advertising.append(AdvertisingEnvironment(s, budgets=daily_budget, sigma=sigma_advertising))
        advanced_ts_learners.append(AdvancedTSLearner(n_arms=n_arms_pricing))
        gpts_learner.append(GPTSLearner(n_arms=n_arms_advertising, arms=daily_budget))
        # add gp learner

    for t in range(0, T):

        values_combination_of_each_subcampaign = []

        # Thompson Sampling and GP-TS Learner
        for s in subcampaigns:
            pulled_arm, conversion_rate = advanced_ts_learners[s].pull_arm()
            reward = environments_pricing[s].round(pulled_arm)
            advanced_ts_learners[s].update(pulled_arm, reward)

            click_numbers_vector = np.array(gpts_learner[s].pull_arm())
            modified_rewards = click_numbers_vector * pulled_arm * conversion_rate
            values_combination_of_each_subcampaign.append(modified_rewards.tolist())

        # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
        # and save the chosen arm of each sub campaign

        superarm = Knapsack(values_combination_of_each_subcampaign, daily_budget).solve()

        # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
        total_clicks = 0
        for s in subcampaigns:
            reward = environments_advertising[s].round(superarm[s])
            total_clicks += reward
            gpts_learner[s].update(superarm[s], reward)

        total_clicks_per_t.append(total_clicks)

    for s in subcampaigns:
        ts_rewards_per_experiment[s].append(advanced_ts_learners[s].collected_rewards)
        gp_rewards_per_experiment[s].append(gpts_learner[s].collected_rewards)

fig, axs = plt.subplots(3, 2, figsize=(14, 8))
for s in subcampaigns:
    # axs[subcampaign, 0].figure("subcampaign" + str(subcampaign) + ".1")
    axs[s, 0].plot(np.cumsum(np.mean(np.array(opt_normalized) - ts_rewards_per_experiment[s], axis=0)), 'r')
    axs[s, 0].legend(["TS", "Greedy"])

    # axs.figure("subcampaign" + str(subcampaign) + ".2")
    axs[s, 1].plot((np.mean(np.array(opt_normalized) - ts_rewards_per_experiment[s], axis=0)), 'r')
    axs[s, 1].legend(["TS", "Greedy"])

for ax in axs.flat:
    if list(axs.flat).index(ax) % 2 == 0:
        ax.set(xlabel='t', ylabel='CumRegret')
    else:
        ax.set(xlabel='t', ylabel='Regret')
    # ax.label_outer()

plt.show()

# Find the optimal value executing the Knapsack optimization on the different environment
# TODO: find the best way to get the optimum value
total_optimal_combination = []
for s in subcampaigns:
    total_optimal_combination.append(environments_advertising[s].means)
optimal_reward = Knapsack(total_optimal_combination, daily_budget).solve()
opt_advertising = 0
for s in subcampaigns:
    opt += environments_advertising[s].means[optimal_reward[s]]

np.set_printoptions(precision=3)
print("Opt")
print(opt_advertising)
print("Rewards")
print(gp_rewards_per_experiment)
print("Regrets")
regrets = np.mean(np.array(opt_advertising) - gp_rewards_per_experiment, axis=0)
print(regrets)
plt.figure()
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(np.array(opt_advertising) - gp_rewards_per_experiment, axis=0)), 'g')
plt.legend(["Cumulative Regret"])
plt.show()

plt.figure()
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot((np.mean(np.array(opt_advertising) - gp_rewards_per_experiment, axis=0)), 'r')
plt.legend(["Regret"])
plt.show()

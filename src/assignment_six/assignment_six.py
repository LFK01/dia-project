import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from src.advertising.learner.gpts_learner import GPTSLearner
from src.advertising.solver.knapsack import Knapsack
from src.assignment_six.advanced_ts_learner import AdvancedTSLearner
from src.pricing.environment import Environment as PricingEnvironment
from src.advertising.environment.click_budget import ClickBudget as AdvertisingEnvironment
from src.pricing.greedy_learner import GreedyLearner
from src.pricing.reward_function import rewards

T = 50

n_experiments = 1

subcampaigns = [0, 1, 2]

min_value_advertising = 0.0
max_value_advertising = 1.0
sigma_advertising = 10

n_arms_advertising = 21

daily_budget = np.linspace(min_value_advertising, max_value_advertising, n_arms_advertising)

min_value_pricing = 0.0
max_value_pricing = 100.0

n_arms_pricing = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

conversion_prices = np.linspace(min_value_pricing, max_value_pricing, n_arms_pricing)
rewards = rewards(conversion_prices, max_value_pricing)
opt_pricing = np.max(rewards)
rewards_normalized = np.divide(rewards, opt_pricing)
opt_pricing_normalized = np.max(rewards_normalized)

ts_rewards_per_experiment_pricing = []
greedy_rewards_per_experiment_pricing = []
gp_rewards_per_experiment_advertising = []

environments_pricing = []
environments_advertising = []

advanced_ts_learners_pricing = []
# greedy_learners_pricing = []
gpts_learner_advertising = []

for s in subcampaigns:
    ts_rewards_per_experiment_pricing.append([])
    # greedy_rewards_per_experiment_pricing.append([])
    environments_pricing.append(PricingEnvironment(n_arms=n_arms_pricing, probabilities=rewards_normalized))
    environments_advertising.append(AdvertisingEnvironment(s, budgets=daily_budget, sigma=sigma_advertising))

for e in range(0, n_experiments):

    advanced_ts_learners_pricing = []
    # greedy_learners_pricing = []
    gpts_learner_advertising = []

    total_revenue_per_t = []

    for s in subcampaigns:
        advanced_ts_learners_pricing.append(AdvancedTSLearner(n_arms=n_arms_pricing, prices=conversion_prices))
        # greedy_learners_pricing.append(GreedyLearner(n_arms=n_arms_pricing, probabilities=[1], number_of_classes=1))
        gpts_learner_advertising.append(GPTSLearner(n_arms=n_arms_advertising, arms=daily_budget))

    description = 'Experiment ' + str(e+1) + ' - Time processed'
    for t in tqdm(range(0, T), desc=description, unit='t'):

        values_combination_of_each_subcampaign = []
        best_price_list = []
        conversion_rate_list = []
        price_index_list = []

        # Thompson Sampling and GP-TS Learner
        for s in subcampaigns:
            # greedy
            # proposed_price = greedy_learners_pricing[s].pull_arm()
            # reward_pricing = environments_pricing[s].round(proposed_price)
            # greedy_learners_pricing[s].update(proposed_price, [reward_pricing])

            # thompson sampling
            price_index, conversion_rate = advanced_ts_learners_pricing[s].pull_arm()
            proposed_price = advanced_ts_learners_pricing[s].get_price_from_index(idx=price_index)
            price_index_list.append(price_index)
            conversion_rate_list.append(conversion_rate)
            best_price_list.append(proposed_price)
            reward_pricing = environments_pricing[s].round(price_index)
            advanced_ts_learners_pricing[s].update(price_index, reward_pricing)

            # advertising
            click_numbers_vector = np.array(gpts_learner_advertising[s].pull_arm())
            modified_rewards = click_numbers_vector * proposed_price * conversion_rate
            values_combination_of_each_subcampaign.append(modified_rewards.tolist())

        # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
        # and save the chosen arm of each sub campaign

        superarm = Knapsack(values_combination_of_each_subcampaign, daily_budget).solve()

        # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
        total_revenue = 0
        for s in subcampaigns:
            reward_advertising = environments_advertising[s].round(superarm[s])
            total_revenue += reward_advertising * best_price_list[s] * environments_pricing[s].probabilities[price_index_list[s]]
            gpts_learner_advertising[s].update(superarm[s], reward_advertising)

        total_revenue_per_t.append(total_revenue)

    for s in subcampaigns:
        ts_rewards_per_experiment_pricing[s].append(advanced_ts_learners_pricing[s].collected_rewards)
        # greedy_rewards_per_experiment_pricing[s].append(greedy_learners_pricing[s].collected_rewards)

    gp_rewards_per_experiment_advertising.append(total_revenue_per_t)

# fig_1, axs_1 = plt.subplots(3, 2, figsize=(14, 8))
# for s in subcampaigns:
#     # axs[subcampaign, 0].figure("subcampaign" + str(subcampaign) + ".1")
#     # cumulative regret
#     axs_1[s, 0].plot(np.cumsum(np.mean(np.array(opt_pricing_normalized)
#                                        - ts_rewards_per_experiment_pricing[s], axis=0)), 'r')
#     axs_1[s, 0].plot(np.cumsum(np.mean(np.array(opt_pricing_normalized)
#                                        - greedy_rewards_per_experiment_pricing[s], axis=0)), 'g')
#     axs_1[s, 0].legend(["TS", "Greedy"])
#
#     # axs.figure("subcampaign" + str(subcampaign) + ".2")
#     # instantaneous regret
#     axs_1[s, 1].plot((np.mean(np.array(opt_pricing_normalized) - ts_rewards_per_experiment_pricing[s], axis=0)), 'r')
#     axs_1[s, 1].plot((np.mean(np.array(opt_pricing_normalized)
#                               - greedy_rewards_per_experiment_pricing[s], axis=0)), 'g')
#     axs_1[s, 1].legend(["TS", "Greedy"])
#
# for ax in axs_1.flat:
#     if list(axs_1.flat).index(ax) % 2 == 0:
#         ax.set(xlabel='t', ylabel='CumRegret')
#     else:
#         ax.set(xlabel='t', ylabel='Regret')
#     # ax.label_outer()
#
# plt.show()

# Find the optimal value executing the Knapsack optimization on the different environment

total_optimal_combination = []

conversion_rate_list = []
best_price_list = []

for s in subcampaigns:
    best_conversion_rate = np.max(environments_pricing[s].probabilities)
    conversion_rate_list.append(best_conversion_rate)
    index_of_best_conversion_rate = np.argwhere(environments_pricing[s].probabilities == best_conversion_rate).flatten()
    best_price = advanced_ts_learners_pricing[s].prices[index_of_best_conversion_rate].flatten()
    best_price_list.append(best_price)
    click_numbers_vector = np.array(environments_advertising[s].means)
    modified_rewards = click_numbers_vector * best_price * best_conversion_rate
    total_optimal_combination.append(modified_rewards.tolist())

optimal_reward = Knapsack(total_optimal_combination, daily_budget).solve()

opt_advertising = 0

for s in subcampaigns:
    opt_advertising += environments_advertising[s].means[optimal_reward[s]] \
                       * conversion_rate_list[s] * best_price_list[s]

np.set_printoptions(precision=3)
print("Opt")
print(opt_advertising)
print("Rewards")
print(gp_rewards_per_experiment_advertising)
print("Regrets")
regrets = np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising, axis=0)
print(regrets)
plt.figure()
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising, axis=0)), 'g')
plt.legend(["Cumulative Regret"])
plt.show()

plt.figure()
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot((np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising, axis=0)), 'r')
plt.legend(["Regret"])
plt.show()

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from src.advertising.learner.gpts_learner import GPTSLearner
from src.advertising.solver.knapsack import Knapsack
from src.pricing.environment import Environment as PricingEnvironment
from src.advertising.environment.click_budget import ClickBudget as AdvertisingEnvironment
from src.pricing.reward_function import rewards
from src.pricing.ts_learner import TSLearner

T = 100

n_experiments = 10

subcampaigns = [0, 1, 2]
user_classes_probabilities_vector = [1 / 4, 1 / 2, 1 / 4]

min_value_advertising = 0.0
max_value_advertising = 1.0
sigma_advertising = 1

n_arms_advertising = 21

daily_budget = np.linspace(min_value_advertising, max_value_advertising, n_arms_advertising)

min_value_pricing = 0.0
max_value_pricing = 100.0

n_arms_pricing = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

conversion_prices = np.linspace(min_value_pricing, max_value_pricing, n_arms_pricing)
rewards = rewards(conversion_prices, max_value_pricing)
opt_pricing = np.max(rewards)
rewards_normalized = np.divide(rewards, opt_pricing)

gp_rewards_per_experiment_advertising = []

environments_pricing = []
environments_advertising = []

ts_learner_pricing = None
gpts_learner_advertising = []

for s in subcampaigns:
    environments_pricing.append(PricingEnvironment(n_arms=n_arms_pricing, probabilities=rewards_normalized))
    environments_advertising.append(AdvertisingEnvironment(s, budgets=daily_budget, sigma=sigma_advertising))

for e in range(0, n_experiments):
    total_revenue_per_arm = []
    for price_index in range(n_arms_pricing):
        ts_learner_pricing = TSLearner(n_arms=n_arms_pricing,
                                       probabilities=user_classes_probabilities_vector,
                                       number_of_classes=len(user_classes_probabilities_vector))
        ts_learner_pricing.set_prices(conversion_prices)

        gpts_learner_advertising = []
        total_revenue_per_t = []

        for s in subcampaigns:
            gpts_learner_advertising.append(GPTSLearner(n_arms=n_arms_advertising, arms=daily_budget))

            # Learning of hyper parameters before starting the algorithm
            new_x = []
            new_y = []
            for i in range(0, 80):
                new_x.append(np.random.choice(daily_budget, 1))
                new_y.append(environments_advertising[s].round(np.where(daily_budget == new_x[i])))
            gpts_learner_advertising[s].generate_gaussian_process(new_x, new_y)

        description = 'Experiment ' + str(e + 1) + ', Arm ' + str(price_index + 1) + ' - Time processed'
        for t in tqdm(range(0, T), desc=description, unit='t'):

            values_combination_of_each_subcampaign = []

            # Thompson Sampling and GP-TS Learner
            conversion_rate_vector = ts_learner_pricing.get_conversion_rate(price_index)
            proposed_price = ts_learner_pricing.get_price_from_index(idx=price_index)

            reward_pricing = []

            for s in subcampaigns:
                reward_pricing.append(environments_pricing[s].round(price_index))
            ts_learner_pricing.update(price_index, reward_pricing)

            for s in subcampaigns:
                # advertising
                click_numbers_vector = np.array(gpts_learner_advertising[s].pull_arm())
                modified_rewards = click_numbers_vector * proposed_price * conversion_rate_vector[s]
                values_combination_of_each_subcampaign.append(modified_rewards.tolist())

            # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
            # and save the chosen arm of each sub campaign

            superarm = Knapsack(values_combination_of_each_subcampaign, daily_budget).solve()

            # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
            total_revenue = 0
            for s in subcampaigns:
                reward_advertising = environments_advertising[s].round(superarm[s])
                total_revenue += reward_advertising * proposed_price * environments_pricing[s].probabilities[price_index]
                gpts_learner_advertising[s].update(superarm[s], reward_advertising)

            total_revenue_per_t.append(total_revenue)
        total_revenue_per_arm.append(total_revenue_per_t)
    gp_rewards_per_experiment_advertising.append(total_revenue_per_arm)

# Find the optimal value executing the Knapsack optimization on the different environment
gp_rewards_per_experiment_advertising = np.array(gp_rewards_per_experiment_advertising)

revenue_advertising_list = []

for conversion_rate_index in range(n_arms_pricing):
    price = ts_learner_pricing.prices[conversion_rate_index]
    conversion_rate_list = []
    total_optimal_combination = []

    for s in subcampaigns:
        conversion_rate_list.append(environments_pricing[s].probabilities[conversion_rate_index])
        click_numbers_vector = np.array(environments_advertising[s].means)
        modified_rewards = click_numbers_vector * price * conversion_rate_list[s]
        total_optimal_combination.append(modified_rewards.tolist())

    optimal_reward = Knapsack(total_optimal_combination, daily_budget).solve()

    revenue_advertising = 0

    for s in subcampaigns:
        revenue_advertising += environments_advertising[s].means[optimal_reward[s]] \
                           * conversion_rate_list[s] * price
    revenue_advertising_list.append(revenue_advertising)

opt_advertising = max(revenue_advertising_list)

for arm in range(n_arms_pricing):
    np.set_printoptions(precision=3)
    print("Opt")
    print(opt_advertising)
    print("Rewards")
    print(np.mean(gp_rewards_per_experiment_advertising[:, arm, :], axis=0))
    print("Regrets")
    regrets = np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising[:, arm, :], axis=0)
    print(regrets)
    plt.figure()
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising[:, arm, :], axis=0)),
             'g')
    plt.legend(["Cumulative Regret"])
    plt.show()
    # plt.savefig('cum_regret_arm_' + str(arm) + '.png')

    plt.figure()
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot((np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising[:, arm, :], axis=0)), 'r')
    plt.legend(["Regret"])
    plt.show()
    # plt.savefig('regret_arm_' + str(arm) + '.png')

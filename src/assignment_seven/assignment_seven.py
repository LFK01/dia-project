import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from src.advertising.learner.gpts_learner import GPTSLearner
from src.advertising.solver.knapsack import Knapsack
from src.pricing.environment import Environment as PricingEnvironment
from src.advertising.environment.click_budget import ClickBudget as AdvertisingEnvironment
from src.pricing.reward_function import rewards
from src.pricing.ts_learner import TSLearner

# number of timesteps
T = 100

# number of experiments
n_experiments = 2

# subcampaigns array
subcampaigns = [0, 1, 2]
# probabilities of a user to correspond to each class
user_classes_probabilities_vector = [1 / 4, 1 / 2, 1 / 4]

# min and max values for the advertising prices
min_value_advertising = 0.0
max_value_advertising = 1.0
sigma_advertising = 1

# number of arms for the advertising task
n_arms_advertising = 21

# arrays of budgets to spend each day
daily_budget = np.linspace(min_value_advertising, max_value_advertising, n_arms_advertising)

# min and max values for the pricing task
min_value_pricing = 0.0
max_value_pricing = 100.0

# computation of the number of arms for the pricing task
n_arms_pricing = int(np.ceil(np.power(np.log2(T) * T, 1 / 4)))

# prices at which to sell the product
conversion_prices = np.linspace(min_value_pricing, max_value_pricing, n_arms_pricing)
# computation of the rewards constituted of the product of price and probability of selling
rewards = rewards(conversion_prices, max_value_pricing)
opt_pricing = np.max(rewards)
rewards_normalized = np.divide(rewards, opt_pricing)

# array to store the rewards of the gaussian process for each experiment
gp_rewards_per_experiment_advertising = []

# arrays to store the environments for the pricing and advertising tasks
environments_pricing = []
environments_advertising = []

# learner for the pricing task
ts_learner_pricing = None
# array to store the learners for the advertising tasks
gpts_learner_advertising = []

# initialization of the environments
for s in subcampaigns:
    environments_pricing.append(PricingEnvironment(n_arms=n_arms_pricing, conversion_rates=rewards_normalized))
    environments_advertising.append(AdvertisingEnvironment(s, budgets=daily_budget, sigma=sigma_advertising))

# execution of the experiments
for e in range(0, n_experiments):

    # collection of rewards for each arm corresponding to a price
    total_revenue_per_arm = []

    # iteration over the prices
    for price_index in range(n_arms_pricing):
        # initialization of the Thompson Sampling Learner
        ts_learner_pricing = TSLearner(n_arms=n_arms_pricing,
                                       probabilities=user_classes_probabilities_vector,
                                       number_of_classes=len(user_classes_probabilities_vector))
        # saving of the prices
        ts_learner_pricing.set_prices(conversion_prices)

        # arrays to store the Gaussian Processes Thompson Sampling Learners
        gpts_learner_advertising = []
        # array to collect the revenue for each time step
        total_revenue_per_t = []

        # iteration over the subcampaigns
        for s in subcampaigns:
            # initialization of the Thompson Sampling Learner
            gpts_learner_advertising.append(GPTSLearner(n_arms=n_arms_advertising, arms=daily_budget))

            # Learning of hyper parameters before starting the algorithm
            new_x = []
            new_y = []
            for i in range(0, 80):
                new_x.append(np.random.choice(daily_budget, 1))
                new_y.append(environments_advertising[s].round(np.where(daily_budget == new_x[i])))
            gpts_learner_advertising[s].generate_gaussian_process(new_x, new_y)

        # iteration over the timesteps
        description = 'Experiment ' + str(e + 1) + ', Arm ' + str(price_index + 1) + ' - Time processed'
        for t in tqdm(range(0, T), desc=description, unit='t'):

            # array to store all the vectors of modified rewards composed of the product of clicks, price and
            # conversion rates
            values_combination_of_each_subcampaign = []

            # Thompson Sampling
            # retrieve the conversion rate vector from the learner corresponding to one price index
            conversion_rate_vector = ts_learner_pricing.get_conversion_rate(price_index)
            # retrieve the corresponding price
            proposed_price = ts_learner_pricing.get_price_from_index(idx=price_index)

            reward_pricing = []

            # collect the reward from the environments corresponding to the current price
            for s in subcampaigns:
                reward_pricing.append(environments_pricing[s].round(price_index))
            # update the learner
            ts_learner_pricing.update(price_index, reward_pricing)

            for s in subcampaigns:
                # GP-TS advertising
                # retrieve the number of clicks for each arms according to the learner
                click_numbers_vector = np.array(gpts_learner_advertising[s].pull_arm())
                # compute the rewards composed of the product of clicks, prices and conversion rates
                modified_rewards = click_numbers_vector * proposed_price * conversion_rate_vector[s] \
                                                                         * user_classes_probabilities_vector[s]
                # store the rewards in the values_combination_of_each_subcampaign array
                values_combination_of_each_subcampaign.append(modified_rewards.tolist())

            # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
            # and save the chosen arm of each sub campaign

            superarm = Knapsack(values_combination_of_each_subcampaign, daily_budget).solve()

            # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
            total_revenue = 0
            for s in subcampaigns:
                # collect the reward distributed by each environment
                advertising_obtained_clicks = environments_advertising[s].round(superarm[s])
                # update the collected revenue value
                total_revenue += advertising_obtained_clicks * proposed_price \
                                                             * environments_pricing[s].conversion_rates[price_index] \
                                                             * user_classes_probabilities_vector[s]
                # update the learner
                gpts_learner_advertising[s].update(superarm[s], advertising_obtained_clicks)

            # store the accumulated revenue in the timestep
            total_revenue_per_t.append(total_revenue)
        # store the accumulated revenue for the arm
        total_revenue_per_arm.append(total_revenue_per_t)
    # store the accumulated revenue for the experiment
    gp_rewards_per_experiment_advertising.append(total_revenue_per_arm)

# Find the optimal value executing the Knapsack optimization on the different environment
gp_rewards_per_experiment_advertising = np.array(gp_rewards_per_experiment_advertising)

revenue_advertising_list = []

# iterate over the possible prices
for conversion_rate_index in range(n_arms_pricing):
    # retrieve the actual price given the index
    price = ts_learner_pricing.prices[conversion_rate_index]
    conversion_rate_list = []
    total_optimal_combination = []

    # iterate over the subcampaigns
    for s in subcampaigns:
        # store the conversion rate of the corresponding subcampaign given the conversion rate index
        conversion_rate_list.append(environments_pricing[s].conversion_rates[conversion_rate_index])
        # retrieve the number of clicks for a subcampaign
        click_numbers_vector = np.array(environments_advertising[s].means)
        # compute the rewards for this specific price
        modified_rewards = click_numbers_vector * price * conversion_rate_list[s] * user_classes_probabilities_vector[s]
        # store the rewards array for the subcampaign
        total_optimal_combination.append(modified_rewards.tolist())

    # find the optimum reward corresponding to the price
    optimal_reward = Knapsack(total_optimal_combination, daily_budget).solve()

    revenue_advertising = 0

    # compute the total revenue for all the subcampaigns
    for s in subcampaigns:
        revenue_advertising += environments_advertising[s].means[optimal_reward[s]] \
                               * conversion_rate_list[s] * price * user_classes_probabilities_vector[s]
    # store the computed revenue
    revenue_advertising_list.append(revenue_advertising)

# extract the optimal revenue
opt_advertising = max(revenue_advertising_list)

# plot the graphs
fig, axs = plt.subplots(n_arms_pricing, 2)

for arm in range(n_arms_pricing):
    np.set_printoptions(precision=3)
    print("Opt")
    print(opt_advertising)
    print("Rewards")
    print(np.mean(gp_rewards_per_experiment_advertising[:, arm, :], axis=0))
    print("Regrets")
    regrets = np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising[:, arm, :], axis=0)
    print(regrets)
    # axs[arm - 1, 0].ylabel("Regret")
    # axs[arm - 1, 0].xlabel("t")
    axs[arm, 0].plot(np.cumsum(np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising[:, arm, :], axis=0)),
             'g')
    axs[arm, 0].legend(["Cumulative Regret"])
    # plt.savefig('cum_regret_arm_' + str(arm) + '.png')

    # axs[arm - 1, 1].ylabel("Regret")
    # axs[arm - 1, 1].xlabel("t")
    axs[arm, 1].plot((np.mean(np.array(opt_advertising) - gp_rewards_per_experiment_advertising[:, arm, :], axis=0)), 'r')
    axs[arm, 1].legend(["Regret"])

for ax in axs.flat:
    ax.set(xlabel='timesteps', ylabel='')

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

plt.show()
# plt.savefig('regret_arm_' + str(arm) + '.png')

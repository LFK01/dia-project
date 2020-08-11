import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters initialization
from src.assignment_2.click_env import ClickEnv
from src.assignment_2.gpts_learner import GPTSLearner
from src.utils.knapsack import Knapsack

subcampaign = [0, 1, 2]

min_budget = 0.0
max_budget = 1.0
n_arms = 21
daily_budget = np.linspace(min_budget, max_budget, n_arms)
sigma = 3

# Time horizon
T = 250
# Number of experiments
n_experiments = 50
x_values = [np.linspace(min_budget, max_budget, 21) for i in range(0, len(subcampaign))]
# The values of the y for each function
y_values = [np.array([0, 3, 3, 6, 9, 15, 24, 39, 63, 165, 228, 328, 368, 398, 418, 428, 432, 433, 433, 433, 433]),
            np.array([0, 2, 2, 4, 6, 10, 16, 26, 42, 68, 110, 178, 288, 408, 493, 533, 545, 550, 552, 553, 553]),
            np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 527, 590, 605, 608, 609, 610])]

collected_rewards_per_experiments = []
env = []
budgets = []
for s in subcampaign:
    env.append(ClickEnv(daily_budget, sigma, x_values[s], y_values[s], s + 1))

# print("Starting experiments...")
for e in tqdm(range(0, n_experiments), desc="Experiment processed", unit="exp"):
    # Initialize the environment, learner and click for each experiment
    gpts_learner = []
    total_clicks_per_t = []
    for s in subcampaign:
        gpts_learner.append(GPTSLearner(n_arms=n_arms, arms=daily_budget))

        # Learning of hyperparameters before starting the algorithm
        new_x = []
        new_y = []
        for i in range(0, 10):
            for arm in daily_budget:
                new_x.append(arm)
                new_y.append(env[s].round(np.where(daily_budget == arm)))
        gpts_learner[s].generate_gaussian_process(new_x, new_y)

    # For each t in the time horizon, run the GP_TS algorithm
    for t in range(0, T):
        total_subcampaign_combination = []

        for s in subcampaign:
            total_subcampaign_combination.append(gpts_learner[s].pull_arm())

        # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
        # and save the chosen arm of each sub campaign
        superarm = Knapsack(total_subcampaign_combination, daily_budget).solve()

        # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
        total_clicks = 0
        for s in subcampaign:
            reward = env[s].round(superarm[s])
            total_clicks += reward
            gpts_learner[s].update(superarm[s], reward)

        # this  is just to debugging purposes and nothing more
        if total_clicks > 613.0 and 613.0 < env[0].means[superarm[0]] + env[1].means[
            superarm[1]] + env[2].means[superarm[2]]:
            print("\nbudget= ", daily_budget[superarm[0]], "+", daily_budget[superarm[1]], "+",
                  daily_budget[superarm[2]], '=',
                  daily_budget[superarm[0]] + daily_budget[superarm[1]] + daily_budget[superarm[2]], "\n")
            print("\nclicks: ", env[0].means[superarm[0]] + env[1].means[
                superarm[1]] + env[2].means[superarm[2]], "\n")

        # append the clicks to total_clicks
        total_clicks_per_t.append(total_clicks)

    # At the end of each experiment, save the total click of each t of this experiment
    collected_rewards_per_experiments.append(total_clicks_per_t)
    time.sleep(0.01)

# Find the optimal value executing the Knapsack optimization on the different environment
# TODO: find the best way to get the optimum value
total_optimal_combination = []
for s in subcampaign:
    total_optimal_combination.append(env[s].means)
optimal_reward = Knapsack(total_optimal_combination, daily_budget).solve()
opt = 0
for s in subcampaign:
    opt += env[s].means[optimal_reward[s]]

np.set_printoptions(precision=3)
print("Opt")
print(opt)
print("Rewards")
print(collected_rewards_per_experiments)
print("Regrets")
regrets = np.mean(np.array(opt) - collected_rewards_per_experiments, axis=0)
print(regrets)
plt.figure()
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(np.array(opt) - collected_rewards_per_experiments, axis=0)), 'g')
plt.legend(["Cumulative Regret"])
plt.show()

plt.figure()
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot((np.mean(np.array(opt) - collected_rewards_per_experiments, axis=0)), 'r')
plt.legend(["Regret"])
plt.show()

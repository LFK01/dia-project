import numpy as np
import matplotlib.pyplot as plt
from pythonCode.mab.Environment.ClickBudget import *
from pythonCode.mab.GPTS_Learner import *
from pythonCode.mab.Knapsack import *
from pythonCode.mab.SuperArmConstraintSolver import *

subcampaign = [0, 1, 2]

min_budget = 0.0
max_budget = 1.0
n_arms = 11
daily_budget = np.linspace(min_budget, max_budget, n_arms)
sigma = 10

T = 10

n_experiments = 2

collected_rewards_per_experiments = []

for e in range(0, n_experiments):
    # Initialize the environment, learner and click for each experiment
    env = []
    gpts_learner = []
    total_clicks_per_t = []

    for s in subcampaign:
        env.append(ClickBudget(s, budgets=daily_budget, sigma=sigma))
        gpts_learner.append(GPTS_Learner(n_arms=n_arms, arms=daily_budget))
        # for arm in range(0, n_arms):
        #     x = np.random.choice(daily_budget, 1)
        #     y = env[s].generate_observations(x, noise_std=sigma)
        #     gpts_learner[s].generate_gaussian_process(x, y)

    # For each t in the time horizon, run the GP_TS algorithm
    for t in range(0, T):
        subcampaign_combination = []
        for s in subcampaign:
            pulled_arm = gpts_learner[s].pull_arm()
            reward = env[s].round(pulled_arm)
            gpts_learner[s].update(pulled_arm, reward)

            for idx in range(0, n_arms):
                subcampaign_combination.append(gpts_learner[s].get_predicted_arm(idx))

        # At the and of the GP_TS algorithm of all the sub campaign , run the Knapsack optimization
        # and save the chosen budget of each sub campaign
        budgets = []
        for n in subcampaign:
            for i in daily_budget:
                budgets.append(i)
        superarm = SuperArmConstraintSolver(subcampaign_combination, budgets, max_budget,
                                            n_arms).solve()

        # At the end of each t, save the total click
        total_clicks = 0
        for s in subcampaign:
            reward = gpts_learner[s].get_predicted_arm(superarm[s])
            total_clicks += reward
        total_clicks_per_t.append(total_clicks)

    # At the end of each experiment, save the total click of each t of this experiment
    collected_rewards_per_experiments.append(total_clicks_per_t)
    print(collected_rewards_per_experiments)

opt = 0
# Get the opt by exploring the last environment analyzed (maybe not the best solution)
# TODO: find the best way to get the optimum value
for e in env:
    opt += np.max(e.means)

print("Opt")
print(opt)
print("Rewards")
print(collected_rewards_per_experiments)
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - collected_rewards_per_experiments, axis=0)), 'g')
plt.legend(["GPTS"])
plt.show()

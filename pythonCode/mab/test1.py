import numpy as np
import matplotlib.pyplot as plt
from pythonCode.mab.Environment.ClickBudget import *
from pythonCode.mab.GPTS_Learner import *
from pythonCode.mab.Knapsack import *

subcampaign = [0, 1, 2]

min_budget = 0.0
max_budget = 1.0
n_arms = 11
daily_budget = np.linspace(min_budget, max_budget, n_arms)
sigma = 10

T = 10

n_experiments = 10
subcampaign_combination = np.zeros((len(subcampaign), n_arms))
opt = 0
collected_rewards_per_experiments = []

for e in range(0, n_experiments):
    for s in subcampaign:
        env = ClickBudget(s, budgets=daily_budget, sigma=sigma)
        gpts_learner = GPTS_Learner(n_arms=n_arms, arms=daily_budget)
        for t in range(0, T):
            # GP Thompson Sampling
            pulled_arm = gpts_learner.pull_arm()
            reward = env.round(pulled_arm)
            gpts_learner.update(pulled_arm, reward)
        for idx in range(0, n_arms):
            subcampaign_combination[s, idx] = gpts_learner.get_predicted_arm(idx)

    superarm = Knapsack(subcampaign_combination, max_budget).optimize()
    total_clicks = 0
    for s in subcampaign:
        total_clicks += subcampaign_combination[s, superarm[s]]

    collected_rewards_per_experiments.append(total_clicks)

for s in subcampaign:
    env = ClickBudget(s, budgets=daily_budget, sigma=sigma)
    opt += np.max(env.means)

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

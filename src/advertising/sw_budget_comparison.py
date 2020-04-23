import time
import math
import matplotlib.pyplot as plt
from src.advertising.environment.click_budget import *
from src.advertising.learner.gpts_learner import *
from src.advertising.solver.superarm_constraint_solver import *
from src.advertising.solver.knapsack import *
from tqdm import tqdm

# Parameters initialization
subcampaign = [0, 1, 2]

min_budget = 0.0
max_budget = 1.0
n_arms = 10
daily_budget = np.linspace(min_budget, max_budget, n_arms)
sigma = 10

# Time horizon
T = 150
n_phases = 3
# Window size proportional to T
window_size = int(np.sqrt(T)) * 2
# Number of experiments
n_experiments = 10

collected_rewards_per_experiments = []
env = []
budgets = []
opt_per_phases = [0, 0, 0]

for e in tqdm(range(0, n_experiments), desc="Experiment processed", unit="exp"):
    # Initialize the environment, learner and click for each experiment
    env = []
    gpts_learner = []
    total_clicks_per_t = []

    for s in subcampaign:
        gpts_learner.append(GPTSLearner(n_arms=n_arms, arms=daily_budget))

    # For each t in the time horizon, run the GP_TS algorithm
    for t in range(0, T):
        # For every phase, reinitialize the environments
        if t % (T / n_phases) == 0:
            i = int(t / (T / n_phases))
            env = []
            c = 4 * (i + 1)
            d = 3 / (i + 1)
            e = 3 / (i + 1)
            total_optimal_combination = []
            budgets = []
            for n in subcampaign:
                for b in daily_budget:
                    budgets.append(b)
            # Calculate the optimum for every phase
            for s in subcampaign:
                env.append(ClickBudget(s, budgets=daily_budget, sigma=sigma, c=c, d=d * (s + 1), e=e * (s + 1)))
                for idx in range(0, n_arms):
                    total_optimal_combination.append(env[s].means[idx])

            optimal_reward = Knapsack(total_optimal_combination, budgets, n_arms).solve()
            for s in subcampaign:
                opt_per_phases[i] += env[s].means[optimal_reward[s]]

        total_subcampaign_combination = []
        for s in subcampaign:
            for arm in gpts_learner[s].pull_arm():
                total_subcampaign_combination.append(arm)

        # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
        # and save the chosen arm of each sub campaign
        superarm = Knapsack(total_subcampaign_combination, budgets,
                            n_arms).solve()

        # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
        total_clicks = 0
        for s in subcampaign:
            reward = env[s].round(superarm[s])
            total_clicks += reward
            gpts_learner[s].update(superarm[s], reward, window_size=window_size, sw=True)

        total_clicks_per_t.append(total_clicks)

    # At the end of each experiment, save the total click of each t of this experiment
    collected_rewards_per_experiments.append(total_clicks_per_t)
    time.sleep(1)

swts_instantaneous_regret = np.zeros(T)
phases_len = int(T / n_phases)
optimum_per_round = np.zeros(T)

# For each phase, calculate the regret accordingly to the optimum in this particular phase
for i in range(0, n_phases):
    optimum_per_round[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i]
    swts_instantaneous_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(
        collected_rewards_per_experiments, axis=0)[i * phases_len:(i + 1) * phases_len]

np.set_printoptions(precision=3)
print("Opt: ")
print(opt_per_phases)
print("Rewards")
print(collected_rewards_per_experiments)
print("Regret")
print(swts_instantaneous_regret)
plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(collected_rewards_per_experiments, axis=0), 'b')
plt.plot(optimum_per_round, '--k')
plt.legend(["SW-TS", "Optimum"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
plt.legend(["SW-TS"])
plt.show()

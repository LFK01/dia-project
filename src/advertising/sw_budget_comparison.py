import time
import matplotlib.pyplot as plt
from src.advertising.learner.gpts_learner import *
from src.advertising.solver.knapsack import *
from src.advertising.environment.non_stat_click_budg import *
from tqdm import tqdm

# Parameters initialization
subcampaign = [0, 1, 2]
min_budget = 0.0
max_budget = 1.0
n_arms = 21
daily_budget = np.linspace(min_budget, max_budget, n_arms)
sigma = 10

# number of phases
n_phases = 3
# Time horizon multiple of the number of phases
T = n_phases * 100
# Window size proportional to the square root of T and always integer
window_size = int(np.sqrt(T) * 3.4)
# Number of experiments
n_experiments = 2

# BUILD OF THE 3 ENVIRONMENTS. ONE FOR EACH SUBCAMPAIGN. The single environment is not stationary
# define the values of the x
x_values = [np.linspace(min_budget, max_budget, n_arms) for i in range(0, n_phases)]
# The values of the y for each phase
y_values = [[0, 3, 6, 15, 36, 87, 136, 183, 223, 258, 288, 313, 333, 348, 358, 364, 366, 367, 367, 367, 367],
            [0, 3, 6, 15, 36, 87, 88, 95, 112, 124, 135, 145, 145, 175, 200, 230, 250, 270, 300, 340, 367],
            [0, 3, 6, 15, 49, 100, 140, 170, 180, 182, 183, 183, 183, 183, 210, 240, 280, 330, 343, 355, 367]]
# define the length of each phase. The first is T/3 -10, the second T/3 and so on
phase_length = [T / 3 - 10, T / 3, T / 3 + 10]
# define the budgets
budget_matrix = [daily_budget for i in range(0, n_phases)]
# define 3 non stationary environment, one for each subcampaign
environment_first_subcampaign = non_stat_click_env(phase_length, x_values, y_values, sigma, budget_matrix)
environment_second_subcampaign = environment_first_subcampaign
environment_third_subcampaign = environment_first_subcampaign
env = [environment_first_subcampaign, environment_second_subcampaign, environment_third_subcampaign]
# END

collected_rewards_per_experiments = []
sw_collected_rewards_per_experiments = []
opt_per_phases = [0, 0, 0]

budgets = []
for n in subcampaign:
    for b in daily_budget:
        budgets.append(b)

# start of the experiments
for e in tqdm(range(0, n_experiments), desc="Experiment processed", unit="exp"):
    # Initialize the environment, learner and click for each experiment
    gpts_learner = []
    sw_gpts_learner = []
    total_clicks_per_t = []
    sw_total_clicks_per_t = []

    for s in subcampaign:
        gpts_learner.append(GPTSLearner(n_arms=n_arms, arms=daily_budget))
        sw_gpts_learner.append(GPTSLearner(n_arms=n_arms, arms=daily_budget))

    # For each t in the time horizon, run the GP_TS algorithm
    for t in range(0, T):
        total_subcampaign_combination = []
        sw_total_subcampaign_combination = []
        for s in subcampaign:
            for arm in gpts_learner[s].pull_arm():
                total_subcampaign_combination.append(arm)
            for sw_arm in sw_gpts_learner[s].pull_arm():
                sw_total_subcampaign_combination.append(sw_arm)

        # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
        # and save the chosen arm of each sub campaign
        superarm = Knapsack(total_subcampaign_combination, budgets, n_arms).solve()
        sw_superarm = Knapsack(sw_total_subcampaign_combination, budgets, n_arms).solve()

        # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
        total_clicks = 0
        sw_total_clicks = 0
        for s in subcampaign:
            reward = env[s].round(superarm[s], t)
            sw_reward = env[s].round(sw_superarm[s], t)
            total_clicks += reward
            sw_total_clicks += sw_reward
            gpts_learner[s].update(superarm[s], reward)
            sw_gpts_learner[s].update(sw_superarm[s], sw_reward, window_size, True)

        total_clicks_per_t.append(total_clicks)
        sw_total_clicks_per_t.append(sw_total_clicks)

    # At the end of each experiment, save the total click of each t of this experiment
    collected_rewards_per_experiments.append(total_clicks_per_t)
    sw_collected_rewards_per_experiments.append(sw_total_clicks_per_t)
    time.sleep(0.01)

# computation of the optimal budget combination for each abrupt phase
ts_instantaneous_regret = np.zeros(T)
swts_instantaneous_regret = np.zeros(T)
optimum_per_round = np.zeros(T)
total_optimal_combination = []
for p in range(0, n_phases):
    total_optimal_combination = []
    for i in subcampaign:
        for idx in range(0, n_arms):
            total_optimal_combination.append(env[i].means[p][idx])
    optimal_reward = Knapsack(total_optimal_combination, budgets, n_arms).solve()
    for s in subcampaign:
        opt_per_phases[p] += env[s].means[p][optimal_reward[s]]

times_of_change = env[1].change_phases_time.tolist()
times_of_change = [int(i) for i in times_of_change]
phase_length = [int(i) for i in phase_length]
print(times_of_change)
# For each phase, calculate the regret accordingly to the optimum in this particular phase
for i in range(0, n_phases - 1):
    optimum_per_round[times_of_change[i]: times_of_change[i + 1]] = opt_per_phases[i]
    ts_instantaneous_regret[times_of_change[i]: times_of_change[i + 1]] = opt_per_phases[i] - np.mean(
        collected_rewards_per_experiments, axis=0)[times_of_change[i]: times_of_change[i + 1]]
    swts_instantaneous_regret[times_of_change[i]: times_of_change[i + 1]] = opt_per_phases[i] - np.mean(
        sw_collected_rewards_per_experiments, axis=0)[times_of_change[i]: times_of_change[i + 1]]

# the last iteration is done out of the cycle and manually. TODO: FIND A WAY TO PUT THESE LINES INSIDE THE CYCLE ABOVE
optimum_per_round[times_of_change[n_phases - 1]: phase_length[n_phases - 1] + times_of_change[n_phases - 1]] = \
opt_per_phases[n_phases - 1]
ts_instantaneous_regret[times_of_change[n_phases - 1]: phase_length[n_phases - 1] + times_of_change[n_phases - 1]] \
    = opt_per_phases[n_phases - 1] - np.mean(collected_rewards_per_experiments, axis=0)[
                                     times_of_change[n_phases - 1]: phase_length[n_phases - 1] + times_of_change[
                                         n_phases - 1]]
swts_instantaneous_regret[times_of_change[n_phases - 1]: phase_length[n_phases - 1] + times_of_change[n_phases - 1]] = \
opt_per_phases[
    n_phases - 1] - np.mean(
    sw_collected_rewards_per_experiments, axis=0)[
                    times_of_change[n_phases - 1]: phase_length[n_phases - 1] + times_of_change[n_phases - 1]]

# plot the results
print("Opt: ")
np.set_printoptions(precision=3)
print(opt_per_phases)
print("Rewards")
np.set_printoptions(precision=3)
print(collected_rewards_per_experiments)
print("Regret")
np.set_printoptions(precision=3)
print(swts_instantaneous_regret)
plt.figure(1)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(collected_rewards_per_experiments, axis=0), 'g')
plt.plot(np.mean(sw_collected_rewards_per_experiments, axis=0), 'r')
plt.plot(optimum_per_round, '--k')
plt.legend(["TS", "SW-TS", "Optimum"])
plt.show()

plt.figure(2)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(ts_instantaneous_regret), 'g')
plt.plot(np.cumsum(swts_instantaneous_regret), 'r')
plt.legend(["TS", "SW-TS"])
plt.show()

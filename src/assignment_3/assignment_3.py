import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from src.assignment_3.non_stat_click_env import NonStatClickEnv
from src.assignment_2.gpts_learner import GPTSLearner
from src.assignment_3.gpts_learner_sw import GPTSLearnerSW
from src.utils.constants import img_path
from src.utils.knapsack import Knapsack

# Parameters initialization
subcampaign = [0, 1, 2]
readFile = ['../data/subcampaign1.csv', '../data/subcampaign2.csv', '../data/subcampaign3.csv']

# Save array to csv
# for i in subcampaign:
#     pd.DataFrame(value[0]).to_csv(readFile[i], index=None)

min_budget = 0.0
max_budget = 1.0
sigma = 3

y_value = [[], [], []]
data = []

# BUILD OF THE 3 ENVIRONMENTS. ONE FOR EACH SUBCAMPAIGN. The single environment is not stationary
for i in subcampaign:
    # Read environment data from csv file
    data.append(pd.read_csv(readFile[i]))
    for j in range(0, len(data[i].index)):
        # The values of the y for each phase
        y_value[i].append(np.array(data[i].iloc[j]))

n_arms = len(data[0].columns)
daily_budget = np.linspace(min_budget, max_budget, n_arms)
x_values = [np.linspace(min_budget, max_budget, n_arms) for i in range(0, len(subcampaign))]

# number of phases
n_phases = 3
# Time horizon multiple of the number of phases
T = n_phases * 120
# Window size proportional to the square root of T and always integer
window_size = 100
# Number of experiments
n_experiments = 5
# The number of the actual abrupt phase
phase_number = 0

# Length of the phases
phase_length = [T / 3, T / 3, T / 3]
# define the budgets
budget_matrix = [daily_budget for _ in range(0, n_phases)]
# define 3 non stationary environment, one for each subcampaign
environment_first_subcampaign = NonStatClickEnv(phase_length, x_values, y_value[0], sigma, budget_matrix, 1, 'r')
environment_second_subcampaign = NonStatClickEnv(phase_length, x_values, y_value[1], sigma, budget_matrix, 2, 'b')
environment_third_subcampaign = NonStatClickEnv(phase_length, x_values, y_value[2], sigma, budget_matrix, 3, 'g')
env = [environment_first_subcampaign, environment_second_subcampaign, environment_third_subcampaign]
# END

collected_rewards_per_experiments = []
sw_collected_rewards_per_experiments = []
opt_per_phases = [0, 0, 0]

total_clicks = 0
sw_total_clicks = 0
total_clicks_per_t = []
sw_total_clicks_per_t = []

# start of the experiments
for e in tqdm(range(0, n_experiments), desc="Experiment processed", unit="exp"):
    phase_number = 0
    # Initialize the environment, learner and click for each experiment
    gpts_learner = []
    sw_gpts_learner = []
    total_clicks_per_t = []
    sw_total_clicks_per_t = []

    for s in subcampaign:
        gpts_learner.append(GPTSLearner(n_arms=n_arms, arms=daily_budget))
        sw_gpts_learner.append(GPTSLearnerSW(n_arms=n_arms, arms=daily_budget, window_size=window_size))

    # For each t in the time horizon, run the GP_TS algorithm
    for t in range(0, T):
        if t % (T / n_phases) == 0:
            phase_number += 1
            for s in subcampaign:
                # Learning of hyperparameters before starting the algorithm
                new_x = []
                new_y = []
                for i in range(0, 20):
                    index = 0
                    for arm in daily_budget:
                        new_x.append(arm)
                        new_y.append(env[s].round_phase(index, phase_number=phase_number))
                        index += 1
                gpts_learner[s].generate_gaussian_process(new_x, new_y, True)
                sw_gpts_learner[s].generate_gaussian_process(new_x, new_y, True)
        total_subcampaign_combination = []
        sw_total_subcampaign_combination = []
        for s in subcampaign:
            total_subcampaign_combination.append(gpts_learner[s].pull_arm())
            sw_total_subcampaign_combination.append(sw_gpts_learner[s].pull_arm())

        # At the and of the GP_TS algorithm of all the sub campaign, run the Knapsack optimization
        # and save the chosen arm of each sub campaign for TS and SWTS
        superarm = Knapsack(total_subcampaign_combination, daily_budget).solve()
        sw_superarm = Knapsack(sw_total_subcampaign_combination, daily_budget).solve()

        # At the end of each t, save the total click of the arms extracted by the Knapsack optimization
        total_clicks = 0
        sw_total_clicks = 0
        for s in subcampaign:
            # TS
            reward = env[s].round(superarm[s], t)
            total_clicks += reward
            gpts_learner[s].update(superarm[s], reward)

            # SWTS
            sw_reward = env[s].round(sw_superarm[s], t)
            sw_total_clicks += sw_reward
            sw_gpts_learner[s].update(sw_superarm[s], sw_reward)

        total_clicks_per_t.append(total_clicks)
        sw_total_clicks_per_t.append(sw_total_clicks)

    # At the end of each experiment, save the total click of each t of this experiment
    collected_rewards_per_experiments.append(total_clicks_per_t)
    sw_collected_rewards_per_experiments.append(sw_total_clicks_per_t)
    time.sleep(0.01)

# Initialize the instantaneous regrets and the optimum per each round
# computation of the optimal budget combination for each abrupt phase
ts_instantaneous_regret = np.zeros(T)
swts_instantaneous_regret = np.zeros(T)
optimum_per_round = np.zeros(T)

# Calculate the optimum for each round
total_optimal_combination = []
for p in range(0, n_phases):
    total_optimal_combination = []
    for i in subcampaign:
        total_optimal_combination.append(env[i].means[p])

    optimal_reward = Knapsack(total_optimal_combination, daily_budget).solve()
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
swts_instantaneous_regret[
times_of_change[n_phases - 1]: phase_length[n_phases - 1] + times_of_change[n_phases - 1]] = \
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
img_name = "assignment_3_reward.png"
plt.savefig(os.path.join(img_path, img_name))
plt.show()

plt.figure(2)
plt.ylabel("Cumulative Regret")
plt.xlabel("t")
plt.plot(np.cumsum(ts_instantaneous_regret), 'g')
plt.plot(np.cumsum(swts_instantaneous_regret), 'r')
plt.legend(["TS", "SW-TS"])
img_name = "assignment_3_cum_regret.png"
plt.savefig(os.path.join(img_path, img_name))
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from code.mab.Non_Stationary_Environment import *
from code.mab.TS_Learner import *
from code.mab.SWTS_Learner import *


if __name__ == '__main__':

    x1 = np.linspace(0, 10, num=16, endpoint=True)
    x2 = np.linspace(0, 10, num=16, endpoint=True)
    x3 = np.linspace(0, 10, num=16, endpoint=True)

    y1 = np.exp(-x1 ** 2 / 9)
    y2 = np.exp(-x2 ** 2 / 9)
    y3 = np.exp(-x3 ** 2 / 9)

    p1 = np.array([y1[0:4], y1[4:8], y1[8:12], y1[12:16]])
    p2= np.array([y2[0:4], y2[4:8], y2[8:12], y2[12:16]])
    p3 = np.array([y3[0:4], y3[4:8], y3[8:12], y3[12:16]])

    n_arms = 4
    # p = np.array([[0.15, 0.1, 0.2, 0.35], [0.35, 0.32, 0.2, 0.35], [0.5, 0.1, 0.1, 0.15], [0.8, 0.32, 0.1, 0.15]])

    T = 400

    n_experiments = 100
    ts_rewards_per_experiment = []
    swts_rewards_per_experiment = []
    window_size = int(np.sqrt(T))

    for e in range(0, n_experiments):
        ts_env = Non_Stationary_Environment(n_arms, probabilities=p, horizon=T)
        ts_learner = TS_Learner(n_arms)

        swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
        swts_learner = SWTS_Learner(n_arms, window_size)

        for t in range(0, T):
            pulled_arm = ts_learner.pull_arm()
            reward = ts_env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            pulled_arm = swts_learner.pull_arm()
            reward = swts_env.round(pulled_arm)
            swts_learner.update(pulled_arm, reward)

        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        swts_rewards_per_experiment.append(swts_learner.collected_rewards)

    ts_instantaneous_regret = np.zeros(T)
    swts_instantaneous_regret = np.zeros(T)
    n_phases = len(p)
    phases_len = int(T / n_phases)
    opt_per_phases = p.max(axis=1)
    optimum_per_round = np.zeros(T)

    for i in range(0, n_phases):
        optimum_per_round[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i]
        ts_instantaneous_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(
            ts_rewards_per_experiment, axis=0)[i * phases_len:(i + 1) * phases_len]
        swts_instantaneous_regret[i * phases_len: (i + 1) * phases_len] = opt_per_phases[i] - np.mean(
            swts_rewards_per_experiment, axis=0)[i * phases_len:(i + 1) * phases_len]

    plt.figure(0)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
    plt.plot(np.mean(swts_rewards_per_experiment, axis=0), 'b')
    plt.plot(optimum_per_round, '--k')
    plt.legend(["TS", "SW-TS", "Optimum"])
    plt.show()

    plt.figure(1)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
    plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
    plt.legend(["TS", "SW-TS"])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from code.mab.Non_Stationary_Environment import *
from code.mab.TS_Learner import *
from code.mab.SWTS_Learner import *
from code.mab.greedy_learner import *


if __name__ == '__main__':

    x1 = np.linspace(0, 10, num=16, endpoint=True)
    x2 = np.linspace(0, 10, num=16, endpoint=True)
    x3 = np.linspace(0, 10, num=16, endpoint=True)

    y1 = 1 - np.exp(-x1 ** 1.8 / 700)
    y2 = 1 - np.exp(-x2 ** 1.8 / 700)
    y3 = 1 - np.exp(-x3 ** 1.8 / 700)

    # p1 = np.array([y1[0:4], y1[4:8], y1[8:12], y1[12:16]])
    # p2 = np.array([y2[0:4], y2[4:8], y2[8:12], y2[12:16]])
    # p3 = np.array([y3[0:4], y3[4:8], y3[8:12], y3[12:16]])

    p1 = np.array(y1)
    p2 = np.array(y2)
    p3 = np.array(y3)

    n_arms = len(p1)
    # p = np.array([[0.15, 0.1, 0.2, 0.35], [0.35, 0.32, 0.2, 0.35], [0.5, 0.1, 0.1, 0.15], [0.8, 0.32, 0.1, 0.15]])
    opt = (np.max(p1) + np.max(p2) + np.max(p3))/3
    T = 400

    n_experiments = 100
    ts_rewards_per_experiment = []
    gr_rewards_per_experiment = []

    for e in range(0, n_experiments):
        env1 = Environment(n_arms=n_arms, probabilities=p1)
        env2 = Environment(n_arms=n_arms, probabilities=p2)
        env3 = Environment(n_arms=n_arms, probabilities=p3)
        ts_learner = TS_Learner(n_arms=n_arms)
        gr_learner = Greedy_Learner(n_arms=n_arms)
        for t in range(0, T):
            # Thompson Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            reward = (env1.round(pulled_arm) + env2.round(pulled_arm) + env3.round(pulled_arm))/3
            ts_learner.update(pulled_arm, reward)

            # Greedy Learner
            pulled_arm = gr_learner.pull_arm()
            reward = (env1.round(pulled_arm) + env2.round(pulled_arm) + env3.round(pulled_arm))/3
            gr_learner.update(pulled_arm, reward)

        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        gr_rewards_per_experiment.append(gr_learner.collected_rewards)

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment,axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment,axis=0)),'g')
plt.legend(["TS","Greedy"])
plt.show()

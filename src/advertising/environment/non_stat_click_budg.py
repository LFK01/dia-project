from src.advertising.environment.click_budget import *
import numpy as np


class non_stat_click_env:

    # the constructor of the class takes as inputs the length of each phase (array of length equal to the number of phases)
    # ,the parameters associated to the click function of each phase (matrix in which the i_th column contains the
    # parameters of the i-th+1 phase),the variance of the noise, the budgets for each phase (a matrix which contains in
    # each row the budgets associated to each phase)
    def __init__(self, phases_lengths, phases_params, sigma, budgets):

        # initialization of the class parameters
        # number of phases
        self.n_phases = len(phases_lengths)

        # An array containing the times in which each phase will change
        # for example if change_phases_time= [0,3,6,9] this means that the first phase will be from 0 (first position of
        # the vector), the second from 3  (second position of the array) the third from 6 (the third position) and so on
        self.change_phases_time = np.zeros(self.n_phases)

        # compute the array of time instants in which a phase change
        for phase_length_index in reversed(range(0, self.n_phases)):
            for index in reversed(range(0, phase_length_index)):
                self.change_phases_time[phase_length_index] = self.change_phases_time[phase_length_index] + phases_lengths[index]

        # Initialization of the single environments which make up the single non stationary one
        self.environments = []

        # build all the environments needed and add them to the environments array
        for index in range(0, self.n_phases):
            self.environments.append(ClickBudget(index, budgets[index], sigma, phases_params[index][0],
                                                 phases_params[index][1], phases_params[index][2],
                                                 phases_params[index][3]))

    # This function takes as inputs the arm to pull and the current time. The round is given by the right environment
    # which is the one in the right phase
    def round(self, pulled_arm, current_t):
        environmentIndex = 0
        if current_t >= self.change_phases_time[self.n_phases - 1]:
            environmentIndex = self.n_phases - 1
        else:
            while current_t < self.change_phases_time[environmentIndex] or current_t >= self.change_phases_time[environmentIndex + 1]:
                environmentIndex += 1
        return self.environments[environmentIndex].round(pulled_arm)


# This main was used only for testing reasons
if __name__ == "__main__":
    phases_lengths = [3, 3, 3, 3]
    phases_p = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    bud = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    obj = non_stat_click_env(phases_lengths, phases_p, 5, bud)
    obj.round(2, 9)

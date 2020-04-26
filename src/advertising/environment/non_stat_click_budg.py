import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class non_stat_click_env:

    # CONSTRUCTOR OF THE CLASS
    # budgets is a matrix (number of phases x number of arms), phases length is an array in which the i-th number is the
    # length of the i+1 th phase. x_values is an matrix and y_values is also a matrix and they are used to compute
    # the functions for each phase interpolating the points. The interpolation is done interpolating the corresponding
    # x_values and y_values row
    def __init__(self, phases_length, x_values, y_values, sigma, budgets_matrix):

        # number of phases
        self.n_phases = len(phases_length)
        # Functions of each abrupt phases obtained through interpolation
        self.functions = [interpolate.interp1d(x_values[index], y_values[index]) for index in range(0, self.n_phases)]
        for phases_index in range(0, self.n_phases):

            # plot the phase_index+1 function
            plt.figure(phases_index)
            plt.ylabel("Rewards")
            plt.xlabel("arms")
            plt.plot(x_values[phases_index], y_values[phases_index], 'r')
            plt.legend(["Environment function of the abrupt phase"+str(phases_index+1)])
            plt.show()

            # An array containing the times in which each phase will change
            # for example if change_phases_time= [0,3,6,9] this means that the first phase will be from 0 (first position of
            # the vector), the second from 3  (second position of the array) the third from 6 (the third position) and so on
            self.change_phases_time = np.zeros(self.n_phases)

        # compute the array of time instants in which a phase change
        for phase_length_index in reversed(range(0, self.n_phases)):
            for index in reversed(range(0, phase_length_index)):
                self.change_phases_time[phase_length_index] = self.change_phases_time[phase_length_index] + \
                                                                  phases_length[index]
        # compute the sigma array
        self.sigmas = np.ones(len(budgets_matrix[0])) * sigma
        # compute the array of means. Each position is the mean of that phase
        self.means = [self.functions[i](budgets_matrix[i]) for i in range(0, self.n_phases)]

    # FUNCTION: ROUND
    # Given the current time it selects the right environment and it returns the right value
    def round(self, pulled_arm, current_t):
        environmentIndex = 0
        # find the right environment index
        if current_t >= self.change_phases_time[self.n_phases - 1]:
            environmentIndex = self.n_phases - 1
        else:
            while current_t < self.change_phases_time[environmentIndex] or \
                    current_t >= self.change_phases_time[environmentIndex + 1]:
                environmentIndex += 1
        # return the round
        return np.maximum(0, np.random.normal(self.means[environmentIndex][pulled_arm], self.sigmas[pulled_arm]))


# main used just for testing just to see if it works. Just 4 phases with different lengths and every function is
# linear.
if __name__ == "__main__":
    phases_l = [1, 2, 3, 4]
    x_val = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for i in range(0, 4)]
    y_val = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for i in range(0, 4)]
    budgets = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    obj = non_stat_click_env(phases_l, x_val, y_val, 1, budgets)
    print(obj.round(2, 2))

import numpy as np
import matplotlib.pyplot as plt


class curve_visualizer:
    def __init__(self, functions, x_values, subcampaign_number):
        self.functions = functions
        self.color = ['r', 'g', 'b']
        self.n_phases = 3
        self.x_values = x_values
        self.subcampaign_number = subcampaign_number

    def plot_curves(self):
        x_max = np.max(self.x_values)
        x_min = np.min(self.x_values)
        x = np.linspace(x_min, x_max, 100)
        y = [[] for i in range(0, self.n_phases)]
        legend_vector = []
        plt.ylabel("Clicks")
        plt.xlabel("Percentage of allocated daily budget")
        for phase in range(0, self.n_phases):
            y[phase] = [self.functions[phase](i) for i in x]
            plt.plot(x, y[phase], self.color[phase])
            legend_vector.append(
                "Environment function of the subcampaign " + str(self.subcampaign_number) + ", abrupt phase: "
                + str(phase + 1))
        plt.legend(legend_vector)
        plt.show()

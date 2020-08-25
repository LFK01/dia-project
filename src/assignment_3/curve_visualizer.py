import os

import numpy as np
import matplotlib.pyplot as plt
from src.utils.constants import subcampaign_names, abrupt_phases_names, img_path


class CurveVisualizer:
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
        plt.title(subcampaign_names[self.subcampaign_number-1] + ' - Abrupt Phases')
        for phase in range(0, self.n_phases):
            y[phase] = [self.functions[phase](i) for i in x]
            plt.plot(x, y[phase], self.color[phase])
            legend_vector.append("Abrupt phase " + str(phase + 1) + " "
                                 + abrupt_phases_names[phase])
        plt.legend(legend_vector)
        img_name = "subcampaign_" + str(self.subcampaign_number) + "_abrupt_phases.png"
        plt.savefig(os.path.join(img_path, img_name))
        plt.show()

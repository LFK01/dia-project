import numpy as np
from itertools import combinations


class Knapsack:
    def __init__(self, subcampaigns_combination, total_budget):
        self.subcampaigns_combination = subcampaigns_combination
        self.total_budget = 10
        self.daily_budget = np.linspace(0, self.total_budget, self.total_budget + 1, dtype=int)

    def optimize(self):
        n_clicks = 0
        combination = []
        for value in combinations(self.daily_budget, 3):
            tot_sum = 0
            for i in value:
                tot_sum += i
            if tot_sum <= self.total_budget:
                temp_clicks = 0
                for idx, budget in enumerate(value):
                    temp_clicks += self.subcampaigns_combination[idx, budget]
                if temp_clicks > n_clicks:
                    n_clicks = temp_clicks
                    combination = np.asarray(value)

        return combination

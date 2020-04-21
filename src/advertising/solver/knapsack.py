import numpy as np


class Knapsack:

    def __init__(self, rewards, budgets, n_arms):
        # initialize rewards vector
        self.rewards = []
        self.rewards.append(rewards[:n_arms])
        self.rewards.append(rewards[n_arms: 2 * n_arms])
        self.rewards.append(rewards[2 * n_arms:])

        # initialize budgets vector
        self.budgets = []
        self.budgets.append(budgets[: n_arms])
        self.budgets.append(budgets[n_arms: 2 * n_arms])
        self.budgets.append(budgets[2 * n_arms:])

        # initialize dynamic programming table, dimensions: (subcampaigns + 1, budgets)
        self.dp_table = np.zeros((len(self.rewards) + 1, len(self.budgets[0])), dtype=int)
        # initialize table to store the best arm for each subcampaign and each budget
        self.best_arm = np.zeros((len(self.rewards), len(self.budgets[0])), dtype=int)

    def solve(self):
        n_rows = self.dp_table.shape[0]
        n_columns = self.dp_table.shape[1]
        # cycle for each row of the dp_table
        for row in range(1, n_rows):
            # cycle for each row of the dp_table
            for column in range(n_columns):
                # initialize max value for each cell to minus infinity
                max_value = float('-inf')
                # cycle through each column until the current one (included, that's why there is a +1)
                for index in range(column + 1):
                    # the current value is the sum of the subcampaign reward associated to [index] and the value of
                    # the dp table associated to [previous row][column-index]
                    # this way the sum is always equal to the budget expressed by the column
                    current_value = self.rewards[row - 1][index] + self.dp_table[row - 1][column - index]
                    # update max value
                    if current_value > max_value:
                        max_value = current_value
                        # save index of the best arm associated to this subcampaign and budget
                        self.best_arm[row - 1][column] = index
                # update max value in the dp table
                self.dp_table[row][column] = max_value

        # initialize solution vector
        solution = [0, 0, 0]

        # select the column with the highest value in the dp table (first occurrence)
        # sol_3_arm = np.argmax(self.dp_table[3])

        # select the column with the highest value in the dp table (last occurrence)
        sol_3_arm = n_columns - 1 - np.argmax(self.dp_table[3][::-1])
        # assign the arm for the third subcampaign by selecting the arm saved in the best_arm table
        solution[2] = self.best_arm[2][sol_3_arm]

        # the column of the second subcampaing associated to the optimal solution is the one at index
        # [column of third subcampaign - index of third subcampaign]
        sol_2_arm = sol_3_arm - solution[2]
        solution[1] = self.best_arm[1][sol_2_arm]

        # the column of the first subcampaing associated to the optimal solution is the one at index
        # [column of second subcampaign - index of second subcampaign]
        sol_1_arm = sol_2_arm - solution[1]
        solution[0] = self.best_arm[0][sol_1_arm]

        return solution

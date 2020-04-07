from docplex.mp.model import Model
import random


# This is a class to solve the linear programming problem in which we have a some rewards for each subcampaigns. The
# method 'solve' will allow you to find the superarm with the best reward given the constraints The constraints are
# the fact that the sum of all the budget can't be bigger than the maximum budget allowed and that it is possible to
# choose not more than 1 arm for each subcampaign
class SuperArmConstraintSolver:

    # The constraint takes as inputs the rewards of each subcampaign  with their related budgets for each arm and the
    # total budget allowed
    def __init__(self, rewards, budgets, totalBudget, armsPerSubcampaing):
        self.rewardsFirstSubcampaign = rewards[0:armsPerSubcampaing - 1]
        self.rewardsSecondSubcampaign = rewards[armsPerSubcampaing: 2 * armsPerSubcampaing - 1]
        self.rewardsThirdSubcampaign = rewards[2 * armsPerSubcampaing: 3 * armsPerSubcampaing - 1]
        self.budgetsFirstSubcampaign = budgets[0:armsPerSubcampaing - 1]
        self.budgetsSecondSubcampaign = budgets[armsPerSubcampaing: 2 * armsPerSubcampaing - 1]
        self.budgetsThirdSubcampaign = budgets[2 * armsPerSubcampaing: 3 * armsPerSubcampaing - 1]
        self.totalBudget = totalBudget
        self.armsPerSubcampaign = armsPerSubcampaing

    def solve(self):
        # The docplex model to solve the optimization problem and find a feasible and optimal superarm
        mod = Model(name="superArm knapsack")

        # The set upon which iterate over the choices
        I = [i for i in range(0, self.armsPerSubcampaign - 1)]

        # These represent lists of binary variables, each for each subcampaign. If the i_th element of the list is 1 it
        # means that the i_th arm for that campaign has been chosen.
        choice_first_subcampaign = mod.binary_var_list(self.armsPerSubcampaign, name="x")
        choice_second_subcampaign = mod.binary_var_list(self.armsPerSubcampaign, name="y")
        choice_third_subcampaign = mod.binary_var_list(self.armsPerSubcampaign, name="z")

        # Constraint: there must be 3 choices among the 3 subcampaigns
        mod.add_constraint(mod.sum(
            choice_first_subcampaign[i] + choice_second_subcampaign[i] + choice_third_subcampaign[i] for i in I) == 3)

        # These 2 constraints make the superarm consist of 1 "arm" for each subcampaign
        # 3 constraints are not necessary since these ones imply that the third will be chosen for sure
        mod.add_constraint(mod.sum(choice_first_subcampaign[i] for i in I) == 1)
        mod.add_constraint(mod.sum(choice_second_subcampaign[i] for i in I) == 1)

        # This constraint means that the superarm is not such that the budget spent is greater than the maximum allowed (self.totalBudget)
        mod.add_constraint(mod.sum(self.budgetsFirstSubcampaign[i] * choice_first_subcampaign[i] for i in I) +
                           mod.sum(self.budgetsSecondSubcampaign[i] * choice_second_subcampaign[i] for i in I) +
                           mod.sum(self.budgetsThirdSubcampaign[i] * choice_third_subcampaign[i] for i in
                                   I) <= self.totalBudget)

        # This is the maximization of the reward. The reward is obtained in this way:
        mod.maximize(mod.sum(choice_first_subcampaign[i] * self.rewardsFirstSubcampaign[i] for i in I) +
                     mod.sum(choice_second_subcampaign[i] * self.rewardsSecondSubcampaign[i] for i in I) +
                     mod.sum(choice_third_subcampaign[i] * self.rewardsThirdSubcampaign[i] for i in I))
        # Notice that we're adding only the rewards of the chosen 3 arms (each for each subcampaign)

        solver = mod.solve()

        # To find the solution it is necessary to maximize the reward. The feasibility is guaranteed by the constraint above
        solution1 = solver.get_values(choice_first_subcampaign).index(1)
        solution2 = solver.get_values(choice_second_subcampaign).index(1)
        solution3 = solver.get_values(choice_third_subcampaign).index(1)

        results = [solution1, solution2, solution3]
        return results

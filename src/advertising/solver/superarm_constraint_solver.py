from docplex.mp.model import Model


# This is a class to solve the linear programming problem in which we have some rewards for each subcampaigns.
# The method 'solve' allows to find the superarm with the best reward given the constraints.
# The constraints are the following:
# - the sum of all the budget can't be bigger than the maximum budget allowed
# - it is not possible to choose more than one arm for each subcampaign
class SuperArmConstraintSolver:

    # The constraint takes as inputs the rewards of each subcampaign with their related budgets for each arm
    # and the total budget allowed
    def __init__(self, rewards, budgets, total_budget, arms_per_subcampaign):
        self.rewards_first_subcampaign = rewards[:arms_per_subcampaign]
        self.rewards_second_subcampaign = rewards[arms_per_subcampaign: 2 * arms_per_subcampaign]
        self.rewards_third_subcampaign = rewards[2 * arms_per_subcampaign:]
        self.budgets_first_subcampaign = budgets[:arms_per_subcampaign]
        self.budgets_second_subcampaign = budgets[arms_per_subcampaign: 2 * arms_per_subcampaign]
        self.budgets_third_subcampaign = budgets[2 * arms_per_subcampaign:]
        self.total_budget = total_budget
        self.arms_per_subcampaign = arms_per_subcampaign

    def solve(self):
        # The docplex model to solve the optimization problem and find a feasible and optimal superarm
        mod = Model(name="super_arm_knapsack")

        # The set upon which iterate over the choices
        I = [i for i in range(0, self.arms_per_subcampaign - 1)]

        # These represent lists of binary variables, each for each subcampaign.
        # If the i_th element of the list is 1 it means that the i_th arm for that campaign has been chosen.
        choice_first_subcampaign = mod.binary_var_list(self.arms_per_subcampaign, name="budget")
        choice_second_subcampaign = mod.binary_var_list(self.arms_per_subcampaign, name="y")
        choice_third_subcampaign = mod.binary_var_list(self.arms_per_subcampaign, name="z")

        # Constraint: there must be 3 choices among the 3 subcampaigns
        mod.add_constraint(mod.sum(
            choice_first_subcampaign[i] + choice_second_subcampaign[i] + choice_third_subcampaign[i] for i in I) == 3)

        # These 2 constraints make the superarm choose exactly 1 "arm" for each subcampaign
        # 3 constraints are not necessary since these ones imply that the third will be chosen for sure
        mod.add_constraint(mod.sum(choice_first_subcampaign[i] for i in I) == 1)
        mod.add_constraint(mod.sum(choice_second_subcampaign[i] for i in I) == 1)

        # This constraint imposes that the budget spent is not greater
        # than the maximum allowed (self.total_budget)
        mod.add_constraint(mod.sum(self.budgets_first_subcampaign[i] * choice_first_subcampaign[i] for i in I) +
                           mod.sum(self.budgets_second_subcampaign[i] * choice_second_subcampaign[i] for i in I) +
                           mod.sum(self.budgets_third_subcampaign[i] * choice_third_subcampaign[i] for i in I)
                           <= self.total_budget)

        # This is the maximization of the reward. The reward is obtained in this way:
        mod.maximize(mod.sum(choice_first_subcampaign[i] * self.rewards_first_subcampaign[i] for i in I) +
                     mod.sum(choice_second_subcampaign[i] * self.rewards_second_subcampaign[i] for i in I) +
                     mod.sum(choice_third_subcampaign[i] * self.rewards_third_subcampaign[i] for i in I))
        # Notice that we're adding only the rewards of the chosen 3 arms (each for each subcampaign)

        solver = mod.solve()

        # To find the solution it is necessary to maximize the reward.
        # The feasibility is guaranteed by the constraint above
        solution1 = solver.get_values(choice_first_subcampaign).index(1)
        solution2 = solver.get_values(choice_second_subcampaign).index(1)
        solution3 = solver.get_values(choice_third_subcampaign).index(1)

        # Organize the solutions in a 3-element array
        results = [solution1, solution2, solution3]
        return results

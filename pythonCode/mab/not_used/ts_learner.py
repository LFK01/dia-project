from pythonCode.mab.Learner.Learner import Learner
import numpy as np
from operator import itemgetter


class TS_Learner(Learner):
    def __init__(self, n_environments, n_arms, budget, prices):
        super().__init__(n_environments, n_arms)
        # save the total feasible budget
        self.budget = budget
        # initialize the parameters of the beta distribution
        # Example: [[[1. 1.],  ...  [1. 1.],  [1. 1.]]]
        self.beta_parameters = np.ones((n_environments, n_arms, 2))
        # store the matrix of prices for each environment
        # Example: [array([0.        , 0.03448276, ... 0.96551724, 1.        ]),
        #           array([0.        , 0.03448276, ... 0.96551724, 1.        ]),
        #           array([0.        , 0.03448276, ... 0.96551724, 1.        ])]
        self.prices = prices

    def pull_arm(self):
        # list used to store the environment from which an arm has already been selected
        # Example: [env number, ..., env number]
        extracted_environments = []
        # vector in which we store the coordinates of the best found arms.
        # Example: [[environment number, arm number], ... ]
        best_arm_coordinates = []
        # initialize the available budget
        remaining_budget = self.budget
        # iterate until a sufficient number of coordinates has been selected namely one for each environment
        while len(best_arm_coordinates) < len(self.beta_parameters):
            # initialize variable max_extracted_sample used to find the best arm to pull
            max_extracted_sample = 0
            # initialize the array where to store the current optimal coordinates
            optimal_coordinates = [None, None]
            # list used to iterate over the environments
            environment_indexes_list = range(0, len(self.beta_parameters))
            # iterate over the environments
            # lambda function deletes the environments already selected from the list
            for env in filter(lambda el: el not in extracted_environments, environment_indexes_list):
                # list used to iterate over the arms
                arm_indexes_list = range(0, len(self.beta_parameters[env]))
                # iterate over the arms
                for arm in arm_indexes_list:
                    # check if it feasible to select the current arm
                    if remaining_budget >= self.prices[env][arm]:
                        # retrieve a sample from the beta distribution of the corresponding arm
                        sample = np.random.beta(self.beta_parameters[env, arm, 0], self.beta_parameters[env, arm, 1])
                        if sample > max_extracted_sample:
                            # update the max value
                            max_extracted_sample = sample
                            # update the optimal coordinates
                            optimal_coordinates = [env, arm]
            # store the coordinates of the best arm
            best_arm_coordinates.append(optimal_coordinates)
            extracted_environments.append(optimal_coordinates[0])

        # we return the coordinates of the tree best arm we've selected, basically we're returning the superarm
        best_arm_coordinates = sorted(best_arm_coordinates, key=itemgetter(0))
        return best_arm_coordinates

    def update(self, pulled_arm, reward):
        # update the turn number
        self.t += 1
        # update the beta parameters accordingly to the received rewards
        self.update_observations(pulled_arm, reward)
        # iterate over the environments
        for env in range(self.n_environments):
            # update alpha of the corresponding environment and arm
            # (pulled_arm[env][0] is the environment location, pulled_arm[env][1] is the arm location)
            self.beta_parameters[pulled_arm[env][0], pulled_arm[env][1], 0] = \
                self.beta_parameters[pulled_arm[env][0], pulled_arm[env][1], 0] + reward[env]
            # update beta
            self.beta_parameters[pulled_arm[env][0], pulled_arm[env][1], 1] = \
                self.beta_parameters[pulled_arm[env][0], pulled_arm[env][1], 1] + 1.0 - reward[env]

import numpy as np
import matplotlib.pyplot as plt
from scipy import  interpolate


# Conversion rate curve:
# given the budget spent, returns number of clicks
def rewards(conversion_prices, n_of_subcampaign, x_values, y_values):
    # noise = np.random.normal(0, 1, len(price_list))
    noise = 0
    rewards_vector = []
    functions = [interpolate.interp1d(x_values[subcampaign], y_values[subcampaign]) for subcampaign in range(0, n_of_subcampaign)]
    for subcampaign in range(n_of_subcampaign):
        rewards_vector.append(np.array([functions[subcampaign](price)*price for price in conversion_prices]))
        rewards_vector[subcampaign] += noise
        negative_indexes = np.argwhere(rewards_vector[subcampaign] < 0)
        rewards_vector[subcampaign][negative_indexes] = 0.0

    return rewards_vector

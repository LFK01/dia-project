import numpy as np
import matplotlib.pyplot as plt


# Conversion rate curve:
# given the budget spent, returns number of clicks
def rewards(price_list, max_price):
    # noise = np.random.normal(0, 1, len(price_list))
    noise = 0
    rewards_vector = price_list * np.exp(-price_list ** 1.8 / (9 * max_price))
    rewards_vector += noise
    negative_indexes = np.argwhere(rewards_vector < 0)
    rewards_vector[negative_indexes] = 0.0

    return rewards_vector

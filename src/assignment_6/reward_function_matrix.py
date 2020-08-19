import numpy as np
import matplotlib.pyplot as plt


# Conversion rate curve:
# given the budget spent, returns number of clicks
def rewards(conversion_prices, max_price, subcampaigns_number):
    # noise = np.random.normal(0, 1, len(price_list))
    noise = 0
    rewards_vector = []
    for subcampaign in range(subcampaigns_number):
        # PER LUCA:
        # sostituire np.exp(-conversion_prices ** 1.8 / (9 * max_price)) con curva probabilità di
        # conversione per punti
        rewards_vector.append(conversion_prices *
                              np.exp(-conversion_prices ** 1.8 / (9 * max_price)))
        rewards_vector[subcampaign] += noise
        negative_indexes = np.argwhere(rewards_vector[subcampaign] < 0)
        rewards_vector[subcampaign][negative_indexes] = 0.0

    return rewards_vector

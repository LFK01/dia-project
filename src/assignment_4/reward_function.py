import numpy as np
import matplotlib.pyplot as plt


# Conversion rate curve:
# given the budget spent, returns number of clicks
def rewards(price_list, demand_function, subcampaign_number, color='g'):
    # noise = np.random.normal(0, 1, len(price_list))
    max_price = np.max(price_list)
    min_price = np.min(price_list)
    noise = 0
    rewards_vector = price_list * demand_function(price_list)
    rewards_vector += noise
    negative_indexes = np.argwhere(rewards_vector < 0)
    rewards_vector[negative_indexes] = 0.0
    x = np.linspace(min_price, max_price, 100)
    y = [demand_function(x[j]) * x[j] for j in range(0, 100)]
    plt.ylabel("Rewards")
    plt.xlabel("arms")
    plt.plot(x, y, color)
    plt.legend(["expected economic gain curve of the subcampaign " + str(subcampaign_number)])
    plt.show()
    return rewards_vector


if __name__ == '__main__':
    min_value_pricing = 0.0
    max_value_pricing = 100.0

    n_arms_pricing = 1000

    conversion_prices = np.linspace(min_value_pricing, max_value_pricing, n_arms_pricing, endpoint=True)
    rewards_array = []
    for _ in range(3):
        rewards_array.append(rewards(conversion_prices, max_value_pricing))

    fig_1, axs_1 = plt.subplots(3, 1, figsize=(14, 8))

    for i in range(3):
        axs_1[i].plot_curves(conversion_prices, rewards_array[i])

    for ax in axs_1:
        ax.set(xlabel='conversion prices', ylabel='conversion rate')

    plt.show()
